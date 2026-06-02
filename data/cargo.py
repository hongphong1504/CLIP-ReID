# encoding: utf-8

import os
import os.path as osp
import glob

from .bases import BaseImageDataset

import pdb

__all__ = ['CARGO', ]


class CARGO(BaseImageDataset):
    dataset_dir = "CARGO"
    dataset_name = 'cargo'

    def __init__(self, root='datasets', verbose=True, pid_begin=0, **kwargs):
        super(CARGO, self).__init__()
        self.root = root
        self.data_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')

        self._check_before_run()
        self.pid_begin = pid_begin

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def process_dir(self, dir_path, relabel=False):
        img_paths = []
        for cam_index in range(13):
            img_paths = img_paths + glob.glob(osp.join(dir_path, f'Cam{cam_index + 1}', '*.jpg'))

        dataset = []
        pid_container = set()

        # Build pid2label for relabeling
        for img_path in sorted(img_paths):
            pid = int(img_path.split('/')[-1].split('_')[2])
            if pid == -1: continue
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for img_path in sorted(img_paths):
            pid = int(img_path.split('/')[-1].split('_')[2])
            camid = int(img_path.split('/')[-1].split('_')[0][3:])
            viewid = 1 if camid <= 5 else 0
            camid -= 1  # index starts from 0

            if pid == -1: continue

            if relabel: pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, viewid))
        return dataset


class CARGO_AA(BaseImageDataset):
    dataset_dir = "CARGO"
    dataset_name = 'cargo_aa'

    def __init__(self, root='datasets', verbose=True, pid_begin=0, **kwargs):
        super(CARGO_AA, self).__init__()
        self.root = root
        self.data_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')

        self._check_before_run()
        self.pid_begin = pid_begin

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def process_dir(self, dir_path, relabel=False):
        img_paths = []
        for cam_index in range(13):
            img_paths = img_paths + glob.glob(osp.join(dir_path, f'Cam{cam_index + 1}', '*.jpg'))

        dataset = []
        pid_container = set()

        # Build pid2label for relabeling
        for img_path in sorted(img_paths):
            pid = int(img_path.split('/')[-1].split('_')[2])
            camid = int(img_path.split('/')[-1].split('_')[0][3:])
            viewid = 1 if camid <= 5 else 0

            if pid == -1 or viewid == 0: continue

            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for img_path in sorted(img_paths):
            pid = int(img_path.split('/')[-1].split('_')[2])
            camid = int(img_path.split('/')[-1].split('_')[0][3:])
            viewid = 1 if camid <= 5 else 0
            camid -= 1  # index starts from 0

            if pid == -1 or viewid == 0: continue

            if relabel: pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, viewid))
        return dataset


class CARGO_GG(BaseImageDataset):
    dataset_dir = "CARGO"
    dataset_name = 'cargo_gg'

    def __init__(self, root='datasets', verbose=True, pid_begin=0, **kwargs):
        super(CARGO_GG, self).__init__()
        self.root = root
        self.data_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')

        self._check_before_run()
        self.pid_begin = pid_begin

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def process_dir(self, dir_path, relabel=False):
        img_paths = []
        for cam_index in range(13):
            img_paths = img_paths + glob.glob(osp.join(dir_path, f'Cam{cam_index + 1}', '*.jpg'))

        dataset = []
        pid_container = set()

        # Build pid2label for relabeling
        for img_path in sorted(img_paths):
            pid = int(img_path.split('/')[-1].split('_')[2])
            camid = int(img_path.split('/')[-1].split('_')[0][3:])
            viewid = 1 if camid <= 5 else 0

            if pid == -1 or viewid == 1: continue

            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for img_path in sorted(img_paths):
            pid = int(img_path.split('/')[-1].split('_')[2])
            camid = int(img_path.split('/')[-1].split('_')[0][3:])
            viewid = 1 if camid <= 5 else 0
            camid -= 1  # index starts from 0

            if pid == -1 or viewid == 1: continue

            if relabel: pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, viewid))
        return dataset


class CARGO_AG(BaseImageDataset):
    dataset_dir = "CARGO"
    dataset_name = 'cargo_ag'

    def __init__(self, root='datasets', verbose=True, pid_begin=0, **kwargs):
        super(CARGO_AG, self).__init__()
        self.root = root
        self.data_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')

        self._check_before_run()
        self.pid_begin = pid_begin

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False, view='Aerial')
        gallery = self.process_dir(self.gallery_dir, relabel=False, view='Ground')

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def process_dir(self, dir_path, relabel=False, view='all'):
        img_paths = []
        for cam_index in range(13):
            img_paths = img_paths + glob.glob(osp.join(dir_path, f'Cam{cam_index + 1}', '*.jpg'))

        dataset = []
        pid_container = set()

        # Build pid2label for relabeling
        for img_path in sorted(img_paths):
            pid = int(img_path.split('/')[-1].split('_')[2])
            camid = int(img_path.split('/')[-1].split('_')[0][3:])
            viewid = 1 if camid <= 5 else 0

            if pid == -1: continue

            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for img_path in sorted(img_paths):
            pid = int(img_path.split('/')[-1].split('_')[2])
            camid = int(img_path.split('/')[-1].split('_')[0][3:])
            viewid = 'Aerial' if camid <= 5 else 'Ground'
            camid = 1 if camid <= 5 else 2
            camid -= 1  # index starts from 0
            
            if view!='all' and viewid !=view:
                continue
            
            if pid == -1: continue

            viewid = 1 if viewid=='Aerial' else 0
            
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, viewid))
        return dataset

class CARGO_GA(BaseImageDataset):
    dataset_dir = "CARGO"
    dataset_name = 'cargo_ag'

    def __init__(self, root='datasets', verbose=True, pid_begin=0, **kwargs):
        super(CARGO_GA, self).__init__()
        self.root = root
        self.data_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')

        self._check_before_run()
        self.pid_begin = pid_begin

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False, view='Ground')
        gallery = self.process_dir(self.gallery_dir, relabel=False, view='Aerial')

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def process_dir(self, dir_path, relabel=False, view='all'):
        img_paths = []
        for cam_index in range(13):
            img_paths = img_paths + glob.glob(osp.join(dir_path, f'Cam{cam_index + 1}', '*.jpg'))

        dataset = []
        pid_container = set()

        # Build pid2label for relabeling
        for img_path in sorted(img_paths):
            pid = int(img_path.split('/')[-1].split('_')[2])
            camid = int(img_path.split('/')[-1].split('_')[0][3:])
            viewid = 1 if camid <= 5 else 0

            if pid == -1: continue

            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for img_path in sorted(img_paths):
            pid = int(img_path.split('/')[-1].split('_')[2])
            camid = int(img_path.split('/')[-1].split('_')[0][3:])
            viewid = 'Aerial' if camid <= 5 else 'Ground'
            camid = 1 if camid <= 5 else 2
            camid -= 1  # index starts from 0
            
            if view!='all' and viewid !=view:
                continue
            
            if pid == -1: continue

            viewid = 1 if viewid=='Aerial' else 0
            
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, viewid))
        return dataset