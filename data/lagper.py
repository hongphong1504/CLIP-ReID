# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings

from .bases import BaseImageDataset


class LAGPeR(BaseImageDataset):

    dataset_dir = ''
    dataset_name = "lagper"

    def __init__(self, root='datasets', verbose=True, pid_begin=0, **kwargs):
        super(LAGPeR, self).__init__()
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'LAGPeR')

        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"LAGPeR".')

        
        
        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        # self.setting_text = osp.join(self.data_dir, 'exp1_A2G.txt')
        # self.setting_text = osp.join(self.data_dir, 'exp2_G2A.txt')
        # self.setting_text = osp.join(self.data_dir, 'exp3_A2G+.txt')
        # self.setting_text = osp.join(self.data_dir, 'exp4_G2A+.txt')
        self.setting_text = osp.join(self.data_dir, 'exp5_G2A+G.txt')

        required_files = [
            self.data_dir,
            self.train_dir,
            self.setting_text
        ]
        self.check_before_run(required_files)

        self.pid_begin = pid_begin
        
        train = self._process_dir(self.train_dir, relabel=True)
        query, gallery = self.process_setting_txt(self.data_dir,self.setting_text)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '**/*.jpg'))
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')

        dataset = []
        pid_container = set()

        # Build pid2label for relabeling
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())

            if pid == -1:
                continue  # junk images are just ignored

            if camid in [3, 6, 9, 13, 16, 19, 22]: 
                viewid = 1
            else:
                viewid = 0

            if camid > 10: # index start from 0 to 20
                camid -= 2
            else:
                camid -= 1

            if relabel: pid = pid2label[pid] 
            dataset.append((img_path, self.pid_begin + pid, camid, viewid))

        return dataset
    

    def process_setting_txt(self, path, text_path, is_train=True):
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')

        with open(text_path,'r') as f:
            query = []
            gallery = []
            for img_path in f:
                split_p = img_path.split('/')[0]
                split = split_p.split('_')[0]
                img_path = osp.join(path, img_path[:-1])

                pid, camid = map(int, pattern.search(img_path).groups())

                if pid == -1:
                    continue

                if camid in [3, 6, 9, 13, 16, 19, 22]: 
                    viewid = 1
                else:
                    viewid = 0

                if camid > 10: # index start from 0 to 20
                    camid -= 2
                else:
                    camid -= 1


                if split == 'query':
                    query.append((img_path, pid, camid, viewid))
                else:
                    gallery.append((img_path, pid, camid, viewid))

        return query, gallery


 

class LAGPeR_A2G(BaseImageDataset):

    dataset_dir = ''
    dataset_name = "lagper"

    def __init__(self, root='datasets', verbose=True, pid_begin=0, **kwargs):
        super(LAGPeR_A2G, self).__init__()
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'LAGPeR')

        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"LAGPeR".')

        
        
        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        # self.setting_text = osp.join(self.data_dir, 'exp1_A2G.txt')
        # self.setting_text = osp.join(self.data_dir, 'exp2_G2A.txt')
        self.setting_text = osp.join(self.data_dir, 'exp3_A2G+.txt')
        # self.setting_text = osp.join(self.data_dir, 'exp4_G2A+.txt')
        # self.setting_text = osp.join(self.data_dir, 'exp5_G2A+G.txt')

        required_files = [
            self.data_dir,
            self.train_dir,
            self.setting_text
        ]
        self.check_before_run(required_files)

        self.pid_begin = pid_begin
        
        train = self._process_dir(self.train_dir, relabel=True)
        query, gallery = self.process_setting_txt(self.data_dir,self.setting_text)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
        

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '**/*.jpg'))
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')

        dataset = []
        pid_container = set()

        # Build pid2label for relabeling
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())

            if pid == -1:
                continue  # junk images are just ignored

            if camid in [3, 6, 9, 13, 16, 19, 22]: 
                viewid = 1
            else:
                viewid = 0

            if camid > 10: # index start from 0 to 20
                camid -= 2
            else:
                camid -= 1

            if relabel: pid = pid2label[pid] 
            dataset.append((img_path, self.pid_begin + pid, camid, viewid))

        return dataset
    
    def process_setting_txt(self, path, text_path, is_train=True):
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')

        with open(text_path,'r') as f:
            query = []
            gallery = []
            for img_path in f:
                split_p = img_path.split('/')[0]
                split = split_p.split('_')[0]
                img_path = osp.join(path, img_path[:-1])

                pid, camid = map(int, pattern.search(img_path).groups())

                if pid == -1:
                    continue

                if camid in [3, 6, 9, 13, 16, 19, 22]: 
                    viewid = 1
                else:
                    viewid = 0

                if camid > 10: # index start from 0 to 20
                    camid -= 2
                else:
                    camid -= 1


                if split == 'query':
                    query.append((img_path, pid, camid, viewid))
                else:
                    gallery.append((img_path, pid, camid, viewid))

        return query, gallery


class LAGPeR_G2A(BaseImageDataset):
    dataset_dir = ''
    dataset_name = "lagper"

    def __init__(self, root='datasets', verbose=True, pid_begin=0, **kwargs):
        super(LAGPeR_G2A, self).__init__()
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'LAGPeR')

        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"LAGPeR".')

        
        
        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        # self.setting_text = osp.join(self.data_dir, 'exp1_A2G.txt')
        # self.setting_text = osp.join(self.data_dir, 'exp2_G2A.txt')
        # self.setting_text = osp.join(self.data_dir, 'exp3_A2G+.txt')
        self.setting_text = osp.join(self.data_dir, 'exp4_G2A+.txt')
        # self.setting_text = osp.join(self.data_dir, 'exp5_G2A+G.txt')

        required_files = [
            self.data_dir,
            self.train_dir,
            self.setting_text
        ]
        self.check_before_run(required_files)

        self.pid_begin = pid_begin
        
        train = self._process_dir(self.train_dir, relabel=True)
        query, gallery = self.process_setting_txt(self.data_dir,self.setting_text)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '**/*.jpg'))
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')

        dataset = []
        pid_container = set()

        # Build pid2label for relabeling
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())

            if pid == -1:
                continue  # junk images are just ignored

            if camid in [3, 6, 9, 13, 16, 19, 22]: 
                viewid = 1
            else:
                viewid = 0

            if camid > 10: # index start from 0 to 20
                camid -= 2
            else:
                camid -= 1

            if relabel: pid = pid2label[pid] 
            dataset.append((img_path, self.pid_begin + pid, camid, viewid))

        return dataset
    
    def process_setting_txt(self, path, text_path, is_train=True):
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')

        with open(text_path,'r') as f:
            query = []
            gallery = []
            for img_path in f:
                split_p = img_path.split('/')[0]
                split = split_p.split('_')[0]
                img_path = osp.join(path, img_path[:-1])

                pid, camid = map(int, pattern.search(img_path).groups())

                if pid == -1:
                    continue

                if camid in [3, 6, 9, 13, 16, 19, 22]: 
                    viewid = 1
                else:
                    viewid = 0

                if camid > 10: # index start from 0 to 20
                    camid -= 2
                else:
                    camid -= 1


                if split == 'query':
                    query.append((img_path, pid, camid, viewid))
                else:
                    gallery.append((img_path, pid, camid, viewid))

        return query, gallery
    

class LAGPeR_G2AG(BaseImageDataset):

    dataset_dir = ''
    dataset_name = "lagper"

    def __init__(self, root='datasets', verbose=True, pid_begin=0, **kwargs):
        super(LAGPeR_G2AG, self).__init__()
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'LAGPeR')

        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"LAGPeR".')

        
        
        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        # self.setting_text = osp.join(self.data_dir, 'exp1_A2G.txt')
        # self.setting_text = osp.join(self.data_dir, 'exp2_G2A.txt')
        # self.setting_text = osp.join(self.data_dir, 'exp3_A2G+.txt')
        # self.setting_text = osp.join(self.data_dir, 'exp4_G2A+.txt')
        self.setting_text = osp.join(self.data_dir, 'exp5_G2A+G.txt')

        required_files = [
            self.data_dir,
            self.train_dir,
            self.setting_text
        ]
        self.check_before_run(required_files)

        self.pid_begin = pid_begin
        
        train = self._process_dir(self.train_dir, relabel=True)
        query, gallery = self.process_setting_txt(self.data_dir,self.setting_text)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '**/*.jpg'))
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')

        dataset = []
        pid_container = set()

        # Build pid2label for relabeling
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())

            if pid == -1:
                continue  # junk images are just ignored

            if camid in [3, 6, 9, 13, 16, 19, 22]: 
                viewid = 1
            else:
                viewid = 0

            if camid > 10: # index start from 0 to 20
                camid -= 2
            else:
                camid -= 1

            if relabel: pid = pid2label[pid] 
            dataset.append((img_path, self.pid_begin + pid, camid, viewid))

        return dataset
    
    def process_setting_txt(self, path, text_path, is_train=True):
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')

        with open(text_path,'r') as f:
            query = []
            gallery = []
            for img_path in f:
                split_p = img_path.split('/')[0]
                split = split_p.split('_')[0]
                img_path = osp.join(path, img_path[:-1])

                pid, camid = map(int, pattern.search(img_path).groups())

                if pid == -1:
                    continue
                    
                if camid in [3, 6, 9, 13, 16, 19, 22]: 
                    viewid = 1
                else:
                    viewid = 0

                if camid > 10: # index start from 0 to 20
                    camid -= 2
                else:
                    camid -= 1


                if split == 'query':
                    query.append((img_path, pid, camid, viewid))
                else:
                    gallery.append((img_path, pid, camid, viewid))

        return query, gallery