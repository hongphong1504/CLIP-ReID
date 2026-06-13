import glob
import re
import mat4py
import pandas as pd
import torch
import re
import warnings
import os.path as osp

from .bases import BaseImageDataset

class AG_ReID(BaseImageDataset):
    def __init__(self, root='datasets', verbose=True, pid_begin=0, **kwargs):
        super(AG_ReID, self).__init__()
        self.dataset_dir = root
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'AG-ReID')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"AG-ReID".')
        
        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        
        self.query_dir = osp.join(self.data_dir, 'query_all_c0')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test_all_c3')

        # self.query_dir = osp.join(self.data_dir, 'query_all_c3')
        # self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test_all_c0')

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir
        ]

        self.check_before_run(required_files)

        self.pid_begin = pid_begin

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

            
    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern_pid = re.compile(r'P([-\d]+)T([-\d]+)A([-\d]+)')
        pattern_camid = re.compile(r'C([-\d]+)F([-\d]+)')

        dataset = []
        pid_container = set()

        # Build pid2label for relabeling
        for img_path in sorted(img_paths):
            fname = osp.split(img_path)[-1]

            pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
            pid = int(pid_part1 + pid_part2 + pid_part3)
            if pid == -1: continue
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for img_path in sorted(img_paths):
            fname = osp.split(img_path)[-1]

            pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
            pid = int(pid_part1 + pid_part2 + pid_part3)
            if pid == -1: continue

            camid, frameid = pattern_camid.search(fname).groups()
            camid = int(camid)
            if camid: 
                camid = 1
                viewid = 0
            else:
                viewid = 1

            if relabel: pid = pid2label[pid]  
            dataset.append((img_path, self.pid_begin + pid, camid, viewid))

        return dataset
            
class AG_ReID_G2A(BaseImageDataset):
    def __init__(self, root='datasets', verbose=True, pid_begin=0, **kwargs):
        super(AG_ReID_G2A, self).__init__()
        self.dataset_dir = root
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'AG-ReID')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"AG-ReID".')
        
        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        
        # self.query_dir = osp.join(self.data_dir, 'query_all_c0')
        # self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test_all_c3')

        self.query_dir = osp.join(self.data_dir, 'query_all_c3')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test_all_c0')
        
        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir
        ]

        self.check_before_run(required_files)

        self.pid_begin = pid_begin

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)


    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern_pid = re.compile(r'P([-\d]+)T([-\d]+)A([-\d]+)')
        pattern_camid = re.compile(r'C([-\d]+)F([-\d]+)')
        
        dataset = []
        pid_container = set()

        # Build pid2label for relabeling
        for img_path in sorted(img_paths):
            fname = osp.split(img_path)[-1]

            pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
            pid = int(pid_part1 + pid_part2 + pid_part3)
            if pid == -1: continue
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for img_path in sorted(img_paths):
            fname = osp.split(img_path)[-1]

            pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
            pid = int(pid_part1 + pid_part2 + pid_part3)
            if pid == -1: continue

            camid, frameid = pattern_camid.search(fname).groups()
            camid = int(camid)
            if camid: 
                camid = 1
                viewid = 0
            else:
                viewid = 1

            if relabel: pid = pid2label[pid]  
            dataset.append((img_path, self.pid_begin + pid, camid, viewid))

        return dataset
