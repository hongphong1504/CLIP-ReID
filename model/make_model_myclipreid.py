import torch
import torch.nn as nn
import numpy as np
import math

from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_   

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text): 
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x)     
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_final(x).type(self.dtype) 

        # all text tokens
        text_tokens = x      # (B, Lt, Dt)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        eot_feat = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] 

        proj_feat = eot_feat @ self.text_projection

        return text_tokens, eot_feat, proj_feat

class SFM(nn.Module):
    def __init__(self,
                 vision_dim=768,
                 text_dim=512,
                 attn_dim=512,
                 reduction=4,
                 image_token_type='patch',
                 text_token_type='eot'):
        super(SFM, self).__init__()
        self.image_token_type = image_token_type
        self.text_token_type = text_token_type

        # Cross-modal attention
        self.text_proj = nn.Linear(text_dim, attn_dim)
        self.norm = nn.LayerNorm(vision_dim)
        self.image_proj = nn.Linear(vision_dim, attn_dim)
        self.attn_bias = nn.Parameter(torch.zeros(1))

        self.text_proj.apply(weights_init_kaiming)
        self.image_proj.apply(weights_init_kaiming)

        # Dynamic channel weighting
        hidden_dim = vision_dim // reduction
        self.channel_mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, vision_dim),
            nn.Sigmoid()
        )

        self.channel_mlp.apply(weights_init_kaiming)

        # Final refinement FC
        self.refine_fc = nn.Linear(vision_dim, vision_dim)
        self.refine_fc.apply(weights_init_kaiming)

        self.out_norm = nn.LayerNorm(vision_dim)

    def forward(self, image_feat, text_feat):
        """
        image_feat: (B, N, Dv)
        text_feat: (B, Dt) if text_token_type is 'eot', else (B, Nt, Dt)
        """
        dtype = image_feat.dtype
        text_feat = text_feat.to(dtype)

        cls_token = image_feat[:, :1]       # (B,1,Dv)
        patch_feat = image_feat[:, 1:]      # (B,N-1,Dv)

        if self.image_token_type == 'patch':
            visual_tokens = patch_feat
        else: # self.image_token_type == 'all'
            visual_tokens = image_feat

        # Query from text
        if self.text_token_type == 'eot':
            Q = self.text_proj(text_feat).type(dtype).unsqueeze(1)   # (B,1,Da)
        else: # self.text_token_type == 'all'
            Q = self.text_proj(text_feat).type(dtype)                # (B,Lt,Da)

        # Key from visual 
        visual_norm = self.norm(visual_tokens)
        K = self.image_proj(visual_norm)         # (B,Lv,Da)

        # Cross-modal attention
        attn = torch.matmul(Q, K.transpose(-1, -2))   
        attn = attn / math.sqrt(K.shape[-1])
        attn = attn + self.attn_bias

        A = torch.sigmoid(attn)                 
        if self.text_token_type == 'all':
            A = A.mean(dim=1, keepdim=True)

        # Gated image representation
        A_t = A.transpose(1, 2)                 
        X_gated = visual_tokens * A_t            

        # Dynamic channel weighting
        if self.text_token_type == 'eot':
            text_global = text_feat
        else:
            text_global = text_feat.mean(dim=1)
        w_c = self.channel_mlp(text_global)       # (B,Dv)
        w_c = w_c.unsqueeze(1)                  # (B,1,Dv)
        channel_refined = X_gated * w_c         # (B,Np,Dv)
        channel_refined = self.refine_fc(channel_refined)

        # Final output
        X_sfm = X_gated + channel_refined
    
        # Re-attach cls token
        if self.image_token_type == 'patch':
            out = torch.cat([cls_token, X_sfm], dim=1)
        else:
            out = X_sfm

        out = self.out_norm(out)

        return out

class TVF(nn.Module):

    def __init__(
        self,
        vision_dim=768,
        text_dim=512,
        attn_dim=512,
        reduction=4,
        image_token_type='patch',
        text_token_type='eot'
    ):
        super().__init__()

        self.image_token_type = image_token_type
        self.text_token_type = text_token_type

        # Cross attention
        self.q_proj = nn.Linear(vision_dim, attn_dim)
        self.k_proj = nn.Linear(text_dim, attn_dim)
        self.v_proj = nn.Linear(text_dim, attn_dim)

        self.out_proj = nn.Linear(attn_dim, vision_dim)

        self.q_proj.apply(weights_init_kaiming)
        self.k_proj.apply(weights_init_kaiming)
        self.v_proj.apply(weights_init_kaiming)
        self.out_proj.apply(weights_init_kaiming)

        # learnable fusion strength
        self.gamma = nn.Parameter(torch.tensor(0.05))

        # channel modulation branch
        hidden_dim = vision_dim // reduction
        self.channel_mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, vision_dim),
            nn.Sigmoid()
        )

        self.channel_mlp.apply(weights_init_kaiming)

        self.refine_fc = nn.Sequential(
            nn.Linear(vision_dim, vision_dim),
            nn.GELU()
        )
        self.refine_fc.apply(weights_init_kaiming)

        self.out_norm = nn.LayerNorm(vision_dim)

    def forward(self, image_feat, text_feat):
        """
        image_feat : (B,Nv,Dv)
        text_feat: (B, Dt) if text_token_type is 'eot', else (B, Nt, Dt)
        """
        dtype = image_feat.dtype
        text_feat = text_feat.to(dtype)

        cls_token = image_feat[:, :1]
        patch_tokens = image_feat[:, 1:]

        if self.image_token_type == 'patch':
            visual_tokens = patch_tokens
        else:
            visual_tokens = image_feat

        if self.text_token_type == 'eot':
            K_text = text_feat.unsqueeze(1)
            V_text = text_feat.unsqueeze(1)
            text_global = text_feat
        else:
            K_text = text_feat
            V_text = text_feat
            text_global = text_feat.mean(dim=1)

        Q = self.q_proj(visual_tokens)
        K = self.k_proj(K_text)
        V = self.v_proj(V_text)

        # Cross-attion
        attn = torch.matmul(Q, K.transpose(-1, -2))
        attn = attn / math.sqrt(Q.shape[-1])
        attn = torch.softmax(attn, dim=-1)

        context = torch.matmul(attn, V)
        context = self.out_proj(context)

        fused = visual_tokens + self.gamma * context

        # Channel modulation
        channel_gate = self.channel_mlp(text_global)
        channel_gate = channel_gate.unsqueeze(1)

        fused = fused * channel_gate

        # Refinement
        fused = fused + self.refine_fc(fused)

        if self.image_token_type == 'patch':
            out = torch.cat([cls_token, fused], dim=1)
        else:
            out = fused

        out = self.out_norm(out)

        return out

class ImageEncoder(nn.Module):
    def __init__(self, clip_visual, cfg):
        super().__init__()
        self.visual = clip_visual

        if cfg.MODEL.FUSION_TYPE == "SFM":
            self.fusion = SFM(
                attn_dim=512, 
                reduction=4, 
                image_token_type=cfg.MODEL.FUSION_IMAGE_TOKENS, 
                text_token_type=cfg.MODEL.FUSION_TEXT_TOKENS
            )
        else: # TVF
            self.fusion = TVF(
                attn_dim=512,
                reduction=4,
                image_token_type=cfg.MODEL.FUSION_IMAGE_TOKENS,
                text_token_type=cfg.MODEL.FUSION_TEXT_TOKENS
            )

    def forward(self, image, text, cv_emb=None):
        x = self.visual.conv1(image) 
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)

        if cv_emb is not None:
            x[:,0] = x[:,0] + cv_emb

        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)

        x11 = self.visual.transformer.resblocks[:11](x)

        x11 = x11.permute(1, 0, 2)
        x11 = self.fusion(x11, text)
        x11 = x11.permute(1, 0, 2)

        x12 = self.visual.transformer.resblocks[11](x11) 

        x11 = x11.permute(1, 0, 2)  # LND -> NLD  
        x12 = x12.permute(1, 0, 2)  # LND -> NLD  

        x12 = self.visual.ln_post(x12)  

        return x11, x12 
        

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.fusion_text_tokens = cfg.MODEL.FUSION_TEXT_TOKENS
        self.concat_penult_feat = cfg.TEST.CONCAT_PENULT_FEAT

        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE   

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = ImageEncoder(clip_model.visual, cfg)
        self.text_encoder = TextEncoder(clip_model)

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {} and viewpoint number is : {}'.format(camera_num, view_num))
            print('using SIE_Lambda is : {}'.format(self.sie_coe))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
            print('using SIE_Lambda is : {}'.format(self.sie_coe))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('viewpoint number is : {}'.format(view_num))
            print('using SIE_Lambda is : {}'.format(self.sie_coe))
            
    def forward(self, image = None, caption=None, cam_label= None, view_label=None):
        with torch.no_grad():
            text_tokens, eot_feat, text_proj = self.text_encoder(caption)

        cv_embed = None
        if cam_label != None and view_label != None:
            cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
        elif cam_label != None:
            cv_embed = self.sie_coe * self.cv_embed[cam_label]
        elif view_label != None:
            cv_embed = self.sie_coe * self.cv_embed[view_label]

        if self.fusion_text_tokens == 'eot':
            text_input = eot_feat
        else:
            text_input = text_tokens

        penult_imfeat, imfeat = self.image_encoder(image=image, text=text_input, cv_emb=cv_embed) 
        penult_imfeat = penult_imfeat[:,0]
        imfeat = imfeat[:,0]

        feat = self.bottleneck(imfeat) 
        
        if self.training:
            cls_score = self.classifier(feat)
            return [cls_score], [penult_imfeat, imfeat]
        else:
            out_feat = imfeat if self.neck_feat == 'before' else feat
            if self.concat_penult_feat:
                out_feat = torch.cat([out_feat, penult_imfeat], dim=1)
            return out_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

