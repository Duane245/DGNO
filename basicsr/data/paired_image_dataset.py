from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file,
                                    paired_Depth_paths_from_folder,
                                    paired_Depth4_paths_from_folder,
                                    paired_paths_from_folder_tif,
                                    paired_paths_from_folder_tif_refin,
                                    paired_paths_from_folder_tif_refin_gt,
                                    paired_paths_from_folder_tif_refin_depth)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, paired_random_crop_DP4, random_augmentation, paired_center_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, padding_DP4, imfrombytesDP, img2tensor_tif

import random
import numpy as np
import torch
import cv2

class Dataset_PairedImage(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        if self.opt['phase'] == 'val':
            gt_size = self.opt['crop_size'] if 'crop_size' in self.opt else None
            if gt_size is not None:  # or gt_size > 0:
                # padding
                if gt_size > 0:
                    img_gt, img_lq = padding(img_gt, img_lq, gt_size)
                    img_gt, img_lq = paired_center_crop(img_gt, img_lq, gt_size, scale,
                                                        gt_path)
            
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
    
def color_to_gray(img):
    c_linear = 0.2126*img[:, :, 0] + 0.7152*img[:, :, 1] + 0.07228*img[:, :, 2]
    c_linear_temp = c_linear.copy()

    c_linear_temp[np.where(c_linear <= 0.0031308)] = 12.92 * c_linear[np.where(c_linear <= 0.0031308)]
    c_linear_temp[np.where(c_linear > 0.0031308)] = 1.055 * np.power(c_linear[np.where(c_linear > 0.0031308)], 1.0/2.4) - 0.055

    img[:, :, 0] = c_linear_temp
    img[:, :, 1] = c_linear_temp
    img[:, :, 2] = c_linear_temp

    return img
    
class Dataset_PairedImage1(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage1, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']
            
        self.max_sig = 0.005**(1/2)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            
            # print(img_lq.shape)
            # Noise
            if random.uniform(0, 1) <= 0.05:
                row,col,ch = img_lq.shape
                mean = 0.0
                sigma = random.uniform(0.001, self.max_sig)
                gauss = np.random.normal(mean,sigma,(row,col,ch)).astype(img_lq.dtype)
                gauss = gauss.reshape(row,col,ch)
                img_lq = np.clip(img_lq + gauss, 0.0, 1.0)
                # print(img_lq.shape)
                
            if random.uniform(0, 1) <= 0.3:
                img_lq = color_to_gray(img_lq)
                img_gt = color_to_gray(img_gt)
            
            if random.uniform(0, 1) <= 0.5:
                row,col,ch = img_lq.shape
                scale = random.uniform(max(min(max(128/row + 1e-2, 128/col + 1e-2), 1.0), 0.7), 1.0)
                img_lq = cv2.resize(img_lq, dsize=(int(col*scale), int(row*scale)), interpolation=cv2.INTER_AREA)
                img_lq = cv2.resize(img_lq, (gt_size, gt_size))
                
                img_gt = cv2.resize(img_gt, dsize=(int(col*scale), int(row*scale)), interpolation=cv2.INTER_AREA)
                img_gt = cv2.resize(img_gt, (gt_size, gt_size))
            
            
                
            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        if self.opt['phase'] == 'val':
            gt_size = self.opt['crop_size'] if 'crop_size' in self.opt else None
            if gt_size is not None:  # or gt_size > 0:
                # padding
                if gt_size > 0:
                    img_gt, img_lq = padding(img_gt, img_lq, gt_size)
                    img_gt, img_lq = paired_center_crop(img_gt, img_lq, gt_size, scale,
                                                        gt_path)
            
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
    
class Dataset_PairedImage_tif(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage_tif, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.BBBCw = opt['BBBCw']
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder_tif(
                opt['BBBCw'],
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)
            # print(self.paths[0:12])

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        
        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        
        from tifffile import imread
        from io import BytesIO
        try:
            # img_gt = imfrombytes(img_bytes, float32=True)
            img_gt = imread(BytesIO(img_bytes)).astype('float32')
            img_gt -= img_gt.min()
            img_gt = img_gt / img_gt.max()
            img_gt = np.expand_dims(img_gt, axis = 2)
            # print(img_gt.shape)
            
        except:
            raise Exception("gt path {} not working".format(gt_path))
        # print(gt_path)
        # print(img_gt)
        # print(img_gt.shape)
        # print(img_gt.max())
        
        
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            # img_lq = imfrombytes(img_bytes, float32=True)
            img_lq = imread(BytesIO(img_bytes)).astype('float32')
            img_lq -= img_lq.min()
            img_lq = img_lq / img_lq.max()
            img_lq = np.expand_dims(img_lq, axis = 2)
            # print(img_lq.shape)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # print(lq_path)
        # print(img_lq)
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        if self.opt['phase'] == 'val':
            gt_size = self.opt['crop_size'] if 'crop_size' in self.opt else None
            if gt_size is not None:  # or gt_size > 0:
                # padding
                if gt_size > 0:
                    img_gt, img_lq = padding(img_gt, img_lq, gt_size)
                    img_gt, img_lq = paired_center_crop(img_gt, img_lq, gt_size, scale,
                                                        gt_path)
        
        
        
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor_tif([img_gt, img_lq], float32=True)
        
        # img_gt = img_gt.repeat(3, 1, 1)
        # img_lq = img_lq.repeat(3, 1, 1)
        
        # img_lq = np.expand_dims(img_lq, axis = 0)
        # img_lq = torch.FloatTensor(img_lq)
        # print(img_lq.shape)
        
        
        # img_gt = np.expand_dims(img_gt, axis = 0)
        # img_gt = torch.FloatTensor(img_gt)
        # print(img_gt.shape)
        
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            
        
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': str(lq_path),
            'gt_path': str(gt_path)
        }

    def __len__(self):
        return len(self.paths)


class Dataset_PairedImage_tif_zstack(Dataset_PairedImage_tif):
    """Triplet z-stack variant: returns 3 blur levels sharing the same GT.

    paths layout (built by paired_paths_from_folder_tif): every 3 consecutive
    entries share the same gt_path, with lq_path at increasing blur levels.
    Triplet sample = paths[3i : 3i+3].

    Geometry-preserving: all three lq's and the gt go through the SAME random
    crop and the SAME augmentation flag (so the consistency loss is comparing
    pixels of the identical underlying scene patch).
    """

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        from tifffile import imread
        from io import BytesIO

        scale = self.opt['scale']
        index = index % (len(self.paths) // 3)
        base = index * 3
        triplet = [self.paths[base + k] for k in range(3)]

        gt_path = triplet[0]['gt_path']
        img_gt = imread(BytesIO(self.file_client.get(gt_path, 'gt'))).astype('float32')
        img_gt -= img_gt.min()
        img_gt = img_gt / (img_gt.max() + 1e-8)
        img_gt = np.expand_dims(img_gt, axis=2)

        lqs = []
        lq_paths = []
        for entry in triplet:
            lp = entry['lq_path']
            lq = imread(BytesIO(self.file_client.get(lp, 'lq'))).astype('float32')
            lq -= lq.min()
            lq = lq / (lq.max() + 1e-8)
            lq = np.expand_dims(lq, axis=2)
            lqs.append(lq)
            lq_paths.append(str(lp))

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding (apply to each lq with the gt to keep shapes aligned)
            padded_lqs = []
            for lq in lqs:
                g_pad, lq_pad = padding(img_gt, lq, gt_size)
                padded_lqs.append(lq_pad)
            img_gt = g_pad  # all paddings yield same gt
            lqs = padded_lqs

            # shared random crop across [gt, lq1, lq2, lq3]
            img_gt, lqs = paired_random_crop(img_gt, lqs, gt_size, scale, gt_path)

            # shared geometric augmentation (same flag for all)
            if self.geometric_augs:
                aug = random_augmentation(img_gt, *lqs)
                img_gt = aug[0]
                lqs = list(aug[1:])

        if self.opt['phase'] == 'val':
            gt_size = self.opt['crop_size'] if 'crop_size' in self.opt else None
            if gt_size is not None and gt_size > 0:
                # val path is single-image in eval; here only used if zstack ds
                # is reused for val, fall back to lq[0]
                img_gt, lq0 = padding(img_gt, lqs[0], gt_size)
                img_gt, lq0 = paired_center_crop(img_gt, lq0, gt_size, scale, gt_path)
                lqs = [lq0, lq0, lq0]

        tensors = img2tensor_tif([img_gt] + lqs, float32=True)
        img_gt = tensors[0]
        img_lq1, img_lq2, img_lq3 = tensors[1], tensors[2], tensors[3]

        if self.mean is not None or self.std is not None:
            for t in (img_lq1, img_lq2, img_lq3, img_gt):
                normalize(t, self.mean, self.std, inplace=True)

        return {
            'lq1': img_lq1,
            'lq2': img_lq2,
            'lq3': img_lq3,
            'gt': img_gt,
            'lq1_path': lq_paths[0],
            'lq2_path': lq_paths[1],
            'lq3_path': lq_paths[2],
            'gt_path': str(gt_path),
        }

    def __len__(self):
        return len(self.paths) // 3


class Dataset_PairedImage_tif_mpt(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage_tif_mpt, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.BBBCw = opt['BBBCw']
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder_tif(
                opt['BBBCw'],
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)
            # print(self.paths[0:12])

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

        self.max_sig = 0.005**(1/2)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        
        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        
        from tifffile import imread
        from io import BytesIO
        try:
            # img_gt = imfrombytes(img_bytes, float32=True)
            img_gt = imread(BytesIO(img_bytes)).astype('float32')
            img_gt -= img_gt.min()
            img_gt = img_gt / img_gt.max()
            img_gt = np.expand_dims(img_gt, axis = 2)
            # print(img_gt.shape)
            
        except:
            raise Exception("gt path {} not working".format(gt_path))
        # print(gt_path)
        # print(img_gt)
        # print(img_gt.shape)
        # print(img_gt.max())
        
        
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            # img_lq = imfrombytes(img_bytes, float32=True)
            img_lq = imread(BytesIO(img_bytes)).astype('float32')
            img_lq -= img_lq.min()
            img_lq = img_lq / img_lq.max()
            img_lq = np.expand_dims(img_lq, axis = 2)
            # print(img_lq.shape)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # print(lq_path)
        # print(img_lq)
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            
            # print(img_lq.shape)
            # Noise
            if random.uniform(0, 1) <= 0.05:
                row,col,ch = img_lq.shape
                mean = 0.0
                sigma = random.uniform(0.001, self.max_sig)
                gauss = np.random.normal(mean,sigma,(row,col,ch)).astype(img_lq.dtype)
                gauss = gauss.reshape(row,col,ch)
                img_lq = np.clip(img_lq + gauss, 0.0, 1.0)
                # print(img_lq.shape)
                # print("img_lq dtype:", img_lq.dtype)
            # print(img_lq.shape)
            
            if random.uniform(0, 1) <= 0.5:
                row,col,ch = img_lq.shape
                scale = random.uniform(max(min(max(128/row + 1e-2, 128/col + 1e-2), 1.0), 0.7), 1.0)
                img_lq = cv2.resize(img_lq, dsize=(int(col*scale), int(row*scale)), interpolation=cv2.INTER_AREA)
                img_lq = cv2.resize(img_lq, (gt_size, gt_size))

                if img_lq.ndim == 2:
                    img_lq = img_lq[:, :, None]
                
                img_gt = cv2.resize(img_gt, dsize=(int(col*scale), int(row*scale)), interpolation=cv2.INTER_AREA)
                img_gt = cv2.resize(img_gt, (gt_size, gt_size))

                if img_gt.ndim == 2:
                    img_gt = img_gt[:, :, None]

                # print(img_lq.shape)
            
            # print(img_lq.shape)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        if self.opt['phase'] == 'val':
            gt_size = self.opt['crop_size'] if 'crop_size' in self.opt else None
            if gt_size is not None:  # or gt_size > 0:
                # padding
                if gt_size > 0:
                    img_gt, img_lq = padding(img_gt, img_lq, gt_size)
                    img_gt, img_lq = paired_center_crop(img_gt, img_lq, gt_size, scale,
                                                        gt_path)
        
        
        
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor_tif([img_gt, img_lq], float32=True)
        
        # img_gt = img_gt.repeat(3, 1, 1)
        # img_lq = img_lq.repeat(3, 1, 1)
        
        # img_lq = np.expand_dims(img_lq, axis = 0)
        # img_lq = torch.FloatTensor(img_lq)
        # print(img_lq.shape)
        
        
        # img_gt = np.expand_dims(img_gt, axis = 0)
        # img_gt = torch.FloatTensor(img_gt)
        # print(img_gt.shape)
        
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            
        
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': str(lq_path),
            'gt_path': str(gt_path)
        }

    def __len__(self):
        return len(self.paths)

import pandas as pd
import os  
class Dataset_PairedImage_tif_prompt(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage_tif_prompt, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.BBBCw = opt['BBBCw']
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        
        
        ### load csv
        csv_files = opt['dataroot_csv']
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            dfs.append(df)  
        self.dfss = pd.concat(dfs, ignore_index=True)
        
        #############################
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder_tif(
                opt['BBBCw'],
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)
            # print(self.paths[0:12])

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        
        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        
        from tifffile import imread
        from io import BytesIO
        try:
            # img_gt = imfrombytes(img_bytes, float32=True)
            img_gt = imread(BytesIO(img_bytes)).astype('float32')
            img_gt -= img_gt.min()
            img_gt = img_gt / img_gt.max()
            img_gt = np.expand_dims(img_gt, axis = 2)
            # print(img_gt.shape)
            
        except:
            raise Exception("gt path {} not working".format(gt_path))
        # print(gt_path)
        # print(img_gt)
        # print(img_gt.shape)
        # print(img_gt.max())
        
        
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            # img_lq = imfrombytes(img_bytes, float32=True)
            img_lq = imread(BytesIO(img_bytes)).astype('float32')
            img_lq -= img_lq.min()
            img_lq = img_lq / img_lq.max()
            img_lq = np.expand_dims(img_lq, axis = 2)
            # print(img_lq.shape)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        
        # print(lq_path)
        filename = os.path.basename(lq_path)
        match = self.dfss[self.dfss['filename'] == filename]
        
        if not match.empty:
            description = match.iloc[0]['description']
            # print(f"✅ 找到描述: {description}")
        else:
            print(f"❌ 未在 CSV 中找到文件名: {filename}")
        
        
        
        # print(img_lq)
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        if self.opt['phase'] == 'val':
            gt_size = self.opt['crop_size'] if 'crop_size' in self.opt else None
            if gt_size is not None:  # or gt_size > 0:
                # padding
                if gt_size > 0:
                    img_gt, img_lq = padding(img_gt, img_lq, gt_size)
                    img_gt, img_lq = paired_center_crop(img_gt, img_lq, gt_size, scale,
                                                        gt_path)
        
        
        
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor_tif([img_gt, img_lq], float32=True)
        
        # img_gt = img_gt.repeat(3, 1, 1)
        # img_lq = img_lq.repeat(3, 1, 1)
        
        # img_lq = np.expand_dims(img_lq, axis = 0)
        # img_lq = torch.FloatTensor(img_lq)
        # print(img_lq.shape)
        
        
        # img_gt = np.expand_dims(img_gt, axis = 0)
        # img_gt = torch.FloatTensor(img_gt)
        # print(img_gt.shape)
        
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            
        
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'prompt': description,
            'lq_path': str(lq_path),
            'gt_path': str(gt_path)
        }

    def __len__(self):
        return len(self.paths)
    
class Dataset_PairedImage_tif_refin(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage_tif_refin, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.BBBCw = opt['BBBCw']
        self.gt_folder, self.lq_folder, self.lq_depth_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_lq_d']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder_tif_refin(
                opt['BBBCw'],
                [self.lq_folder, self.gt_folder,self.lq_depth_folder], ['lq', 'gt', 'lq_d'],
                self.filename_tmpl)
            # print(self.paths[0:12])

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        
        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        
        from tifffile import imread
        from io import BytesIO
        try:
            # img_gt = imfrombytes(img_bytes, float32=True)
            img_gt = imread(BytesIO(img_bytes)).astype('float32')
            img_gt -= img_gt.min()
            img_gt = img_gt / img_gt.max()
            img_gt = np.expand_dims(img_gt, axis = 2)
            # print(img_gt.shape)
            
        except:
            raise Exception("gt path {} not working".format(gt_path))
        # print(gt_path)
        # print(img_gt)
        # print(img_gt.shape)
        # print(img_gt.max())
        
        
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            # img_lq = imfrombytes(img_bytes, float32=True)
            img_lq = imread(BytesIO(img_bytes)).astype('float32')
            img_lq -= img_lq.min()
            img_lq = img_lq / img_lq.max()
            img_lq = np.expand_dims(img_lq, axis = 2)
            # print(img_lq.shape)
        except:
            raise Exception("lq path {} not working".format(lq_path))
        
        lq_d_path = self.paths[index]['lq_d_path']
        img_bytes = self.file_client.get(lq_d_path, 'lq_d')
        try:
            # img_lq = imfrombytes(img_bytes, float32=True)
            img_lq_d = imread(BytesIO(img_bytes)).astype('float32')
            img_lq_d -= img_lq_d.min()
            img_lq_d = img_lq_d / img_lq_d.max()
            img_lq_d = np.expand_dims(img_lq_d, axis = 2)
            # print(img_lq.shape)
        except:
            raise Exception("lq path {} not working".format(lq_d_path))

        # print(lq_path)
        # print(img_lq)
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            # img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            img_gt, img_lq, img_lq_d = padding_DP(img_gt, img_lq, img_lq_d, gt_size)

            # random crop
            # img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
            #                                     gt_path)
            img_gt, img_lq, img_lq_d = paired_random_crop_DP(img_gt, img_lq, img_lq_d, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                # img_gt, img_lq = random_augmentation(img_gt, img_lq)
                img_gt, img_lq, img_lq_d = random_augmentation(img_gt, img_lq, img_lq_d)

        if self.opt['phase'] == 'val':
            gt_size = self.opt['crop_size'] if 'crop_size' in self.opt else None
            if gt_size is not None:  # or gt_size > 0:
                # padding
                if gt_size > 0:
                    # img_gt, img_lq = padding(img_gt, img_lq, gt_size)
                    # img_gt, img_lq = paired_center_crop(img_gt, img_lq, gt_size, scale,
                    #                                     gt_path)
                    img_gt, img_lq, img_lq_d = padding_DP(img_gt, img_lq, img_lq_d, gt_size)
                    img_gt, img_lq, img_lq_d = paired_random_crop_DP(img_gt, img_lq, img_lq_d, gt_size, scale,
                                                        gt_path)
        
        
        
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, img_lq_d = img2tensor_tif([img_gt, img_lq, img_lq_d], float32=True)
        
        # img_gt = img_gt.repeat(3, 1, 1)
        # img_lq = img_lq.repeat(3, 1, 1)
        
        # img_lq = np.expand_dims(img_lq, axis = 0)
        # img_lq = torch.FloatTensor(img_lq)
        # print(img_lq.shape)
        
        
        # img_gt = np.expand_dims(img_gt, axis = 0)
        # img_gt = torch.FloatTensor(img_gt)
        # print(img_gt.shape)
        
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_lq_d, self.mean, self.std, inplace=True)
            
        
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_d': img_lq_d,
            'lq_path': str(lq_path),
            'gt_path': str(gt_path),
            'lq_d_path': str(lq_d_path)
        }

    def __len__(self):
        return len(self.paths)
    
class Dataset_PairedImage_tif_refin_gt(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage_tif_refin_gt, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.BBBCw = opt['BBBCw']
        self.gt_folder, self.lq_folder, self.lq_depth_folder, self.lq_depth_gt_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_lq_d'], opt['dataroot_lq_d_gt']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder_tif_refin_gt(
                opt['BBBCw'],
                [self.lq_folder, self.gt_folder,self.lq_depth_folder, self.lq_depth_gt_folder], ['lq', 'gt', 'lq_d', 'lq_d_gt'],
                self.filename_tmpl)
            # print(self.paths[0:12])

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        
        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        
        from tifffile import imread
        from io import BytesIO
        try:
            # img_gt = imfrombytes(img_bytes, float32=True)
            img_gt = imread(BytesIO(img_bytes)).astype('float32')
            img_gt -= img_gt.min()
            img_gt = img_gt / img_gt.max()
            img_gt = np.expand_dims(img_gt, axis = 2)
            # print(img_gt.shape)
            
        except:
            raise Exception("gt path {} not working".format(gt_path))
        # print(gt_path)
        # print(img_gt)
        # print(img_gt.shape)
        # print(img_gt.max())
        
        
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            # img_lq = imfrombytes(img_bytes, float32=True)
            img_lq = imread(BytesIO(img_bytes)).astype('float32')
            img_lq -= img_lq.min()
            img_lq = img_lq / img_lq.max()
            img_lq = np.expand_dims(img_lq, axis = 2)
            # print(img_lq.shape)
        except:
            raise Exception("lq path {} not working".format(lq_path))
        
        lq_d_path = self.paths[index]['lq_d_path']
        img_bytes = self.file_client.get(lq_d_path, 'lq_d')
        try:
            # img_lq = imfrombytes(img_bytes, float32=True)
            img_lq_d = imread(BytesIO(img_bytes)).astype('float32')
            img_lq_d -= img_lq_d.min()
            img_lq_d = img_lq_d / img_lq_d.max()
            img_lq_d = np.expand_dims(img_lq_d, axis = 2)
            # print(img_lq.shape)
        except:
            raise Exception("lq path {} not working".format(lq_d_path))
        
        lq_d_gt_path = self.paths[index]['lq_d_gt_path']
        img_bytes = self.file_client.get(lq_d_gt_path, 'lq_d_gt')
        try:
            img_lq_d_gt = imfrombytes(img_bytes, float32=True)
            # img_lq_d_gt = imread(BytesIO(img_bytes)).astype('float32')
            # img_lq_d_gt -= img_lq_d_gt.min()
            # img_lq_d_gt = img_lq_d_gt / img_lq_d_gt.max()
            # img_lq_d_gt = np.expand_dims(img_lq_d_gt, axis = 2)
            # print(img_lq.shape)
        except:
            raise Exception("lq path {} not working".format(lq_d_gt_path))

        # print(lq_path)
        # print(img_lq)
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            # img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            img_gt, img_lq, img_lq_d, img_lq_d_gt = padding_DP4(img_gt, img_lq, img_lq_d, img_lq_d_gt, gt_size)

            # random crop
            # img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
            #                                     gt_path)
            img_gt, img_lq, img_lq_d, img_lq_d_gt = paired_random_crop_DP4(img_gt, img_lq, img_lq_d, img_lq_d_gt, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                # img_gt, img_lq = random_augmentation(img_gt, img_lq)
                img_gt, img_lq, img_lq_d, img_lq_d_gt = random_augmentation(img_gt, img_lq, img_lq_d, img_lq_d_gt)

        if self.opt['phase'] == 'val':
            gt_size = self.opt['crop_size'] if 'crop_size' in self.opt else None
            if gt_size is not None:  # or gt_size > 0:
                # padding
                if gt_size > 0:
                    # img_gt, img_lq = padding(img_gt, img_lq, gt_size)
                    # img_gt, img_lq = paired_center_crop(img_gt, img_lq, gt_size, scale,
                    #                                     gt_path)
                    img_gt, img_lq, img_lq_d, img_lq_d_gt = padding_DP4(img_gt, img_lq, img_lq_d, img_lq_d_gt, gt_size)
                    img_gt, img_lq, img_lq_d, img_lq_d_gt = paired_random_crop_DP4(img_gt, img_lq, img_lq_d, img_lq_d_gt, gt_size, scale,
                                                        gt_path)
        
        
        
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, img_lq_d = img2tensor_tif([img_gt, img_lq, img_lq_d], float32=True)
        img_lq_d_gt = img2tensor(img_lq_d_gt, float32=True)
        
        # img_gt = img_gt.repeat(3, 1, 1)
        # img_lq = img_lq.repeat(3, 1, 1)
        
        # img_lq = np.expand_dims(img_lq, axis = 0)
        # img_lq = torch.FloatTensor(img_lq)
        # print(img_lq.shape)
        
        
        # img_gt = np.expand_dims(img_gt, axis = 0)
        # img_gt = torch.FloatTensor(img_gt)
        # print(img_gt.shape)
        
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_lq_d, self.mean, self.std, inplace=True)
            normalize(img_lq_d_gt, self.mean, self.std, inplace=True)
            
        
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_d': img_lq_d,
            'lq_d_gt': img_lq_d_gt,
            'lq_path': str(lq_path),
            'gt_path': str(gt_path),
            'lq_d_path': str(lq_d_path),
            'lq_d_gt_path': str(lq_d_gt_path)
        }

    def __len__(self):
        return len(self.paths)

class Dataset_PairedImage_tif_refin_depth(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage_tif_refin_depth, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.BBBCw = opt['BBBCw']
        self.gt_folder, self.lq_folder, self.lq_depth_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_lq_d']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder_tif_refin_depth(
                opt['BBBCw'],
                [self.lq_folder, self.gt_folder,self.lq_depth_folder], ['lq', 'gt', 'lq_d'],
                self.filename_tmpl)
            # print(self.paths[0:12])

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        
        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        
        from tifffile import imread
        from io import BytesIO
        try:
            # img_gt = imfrombytes(img_bytes, float32=True)
            img_gt = imread(BytesIO(img_bytes)).astype('float32')
            img_gt -= img_gt.min()
            img_gt = img_gt / img_gt.max()
            img_gt = np.expand_dims(img_gt, axis = 2)
            # print(img_gt.shape)
            
        except:
            raise Exception("gt path {} not working".format(gt_path))
        # print(gt_path)
        # print(img_gt)
        # print(img_gt.shape)
        # print(img_gt.max())
        
        
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            # img_lq = imfrombytes(img_bytes, float32=True)
            img_lq = imread(BytesIO(img_bytes)).astype('float32')
            img_lq -= img_lq.min()
            img_lq = img_lq / img_lq.max()
            img_lq = np.expand_dims(img_lq, axis = 2)
            # print(img_lq.shape)
        except:
            raise Exception("lq path {} not working".format(lq_path))
        
        lq_d_path = self.paths[index]['lq_d_path']
        img_bytes = self.file_client.get(lq_d_path, 'lq_d')
        try:
            img_lq_d = imfrombytes(img_bytes, float32=True)
            # img_lq_d = imread(BytesIO(img_bytes)).astype('float32')
            # img_lq_d -= img_lq_d.min()
            # img_lq_d = img_lq_d / img_lq_d.max()
            # img_lq_d = np.expand_dims(img_lq_d, axis = 2)
            # print(img_lq.shape)
        except:
            raise Exception("lq path {} not working".format(lq_d_path))

        # print(lq_path)
        # print(img_lq)
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            # img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            img_gt, img_lq, img_lq_d = padding_DP(img_gt, img_lq, img_lq_d, gt_size)

            # random crop
            # img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
            #                                     gt_path)
            img_gt, img_lq, img_lq_d = paired_random_crop_DP(img_gt, img_lq, img_lq_d, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                # img_gt, img_lq = random_augmentation(img_gt, img_lq)
                img_gt, img_lq, img_lq_d = random_augmentation(img_gt, img_lq, img_lq_d)

        if self.opt['phase'] == 'val':
            gt_size = self.opt['crop_size'] if 'crop_size' in self.opt else None
            if gt_size is not None:  # or gt_size > 0:
                # padding
                if gt_size > 0:
                    # img_gt, img_lq = padding(img_gt, img_lq, gt_size)
                    # img_gt, img_lq = paired_center_crop(img_gt, img_lq, gt_size, scale,
                    #                                     gt_path)
                    img_gt, img_lq, img_lq_d = padding_DP(img_gt, img_lq, img_lq_d, gt_size)
                    img_gt, img_lq, img_lq_d = paired_random_crop_DP(img_gt, img_lq, img_lq_d, gt_size, scale,
                                                        gt_path)
        
        
        
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor_tif([img_gt, img_lq], float32=True)
        img_lq_d = img2tensor(img_lq_d,
                                    bgr2rgb=True,
                                    float32=True)
        
        # img_gt = img_gt.repeat(3, 1, 1)
        # img_lq = img_lq.repeat(3, 1, 1)
        
        # img_lq = np.expand_dims(img_lq, axis = 0)
        # img_lq = torch.FloatTensor(img_lq)
        # print(img_lq.shape)
        
        
        # img_gt = np.expand_dims(img_gt, axis = 0)
        # img_gt = torch.FloatTensor(img_gt)
        # print(img_gt.shape)
        
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_lq_d, self.mean, self.std, inplace=True)
            
        
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_d': img_lq_d,
            'lq_path': str(lq_path),
            'gt_path': str(gt_path),
            'lq_d_path': str(lq_d_path)
        }

    def __len__(self):
        return len(self.paths)
    
class Dataset_GaussianDenoising(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_GaussianDenoising, self).__init__()
        self.opt = opt

        if self.opt['phase'] == 'train':
            self.sigma_type  = opt['sigma_type']
            self.sigma_range = opt['sigma_range']
            assert self.sigma_type in ['constant', 'random', 'choice']
        else:
            self.sigma_test = opt['sigma_test']
        self.in_ch = opt['in_ch']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None        

        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [
                    osp.join(self.gt_folder,
                             line.split(' ')[0]) for line in fin
                ]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')

        if self.in_ch == 3:
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        else:
            try:
                img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = img_gt.copy()


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=False,
                                        float32=True)


            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)

            noise_level = torch.FloatTensor([sigma_value])/255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)

        else:            
            np.random.seed(seed=0)
            img_lq += np.random.normal(0, self.sigma_test/255.0, img_lq.shape)
            # noise_level_map = torch.ones((1, img_lq.shape[0], img_lq.shape[1])).mul_(self.sigma_test/255.0).float()

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                            bgr2rgb=False,
                            float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

class Dataset_DefocusDeblur_DualPixel_16bit(data.Dataset):
    def __init__(self, opt):
        super(Dataset_DefocusDeblur_DualPixel_16bit, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lqL_folder, self.lqR_folder = opt['dataroot_gt'], opt['dataroot_lqL'], opt['dataroot_lqR']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_DP_paths_from_folder(
            [self.lqL_folder, self.lqR_folder, self.gt_folder], ['lqL', 'lqR', 'gt'],
            self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lqL_path = self.paths[index]['lqL_path']
        img_bytes = self.file_client.get(lqL_path, 'lqL')
        try:
            img_lqL = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqL path {} not working".format(lqL_path))

        lqR_path = self.paths[index]['lqR_path']
        img_bytes = self.file_client.get(lqR_path, 'lqR')
        try:
            img_lqR = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqR path {} not working".format(lqR_path))


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_lqL, img_lqR, img_gt = padding_DP(img_lqL, img_lqR, img_gt, gt_size)

            # random crop
            img_lqL, img_lqR, img_gt = paired_random_crop_DP(img_lqL, img_lqR, img_gt, gt_size, scale, gt_path)
            
            # flip, rotation            
            if self.geometric_augs:
                img_lqL, img_lqR, img_gt = random_augmentation(img_lqL, img_lqR, img_gt)
        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lqL, img_lqR, img_gt = img2tensor([img_lqL, img_lqR, img_gt],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lqL, self.mean, self.std, inplace=True)
            normalize(img_lqR, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        img_lq = torch.cat([img_lqL, img_lqR], 0)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lqL_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
    
    
    

class Dataset_PairedImage_depth(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage_depth, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_folder, self.lq_depth_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_lq_depth']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_Depth_paths_from_folder(
                [self.lq_folder, self.gt_folder,self.lq_depth_folder], ['lq', 'gt', 'lq_d'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # index = 0
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))
        
        # print(gt_path)

        lq_path = self.paths[index]['lq_path']
        # print(lq_path)
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))
        
        lq_d_path = self.paths[index]['lq_d_path']
        img_bytes = self.file_client.get(lq_d_path, 'lq_d')
        try:
            img_lq_d = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_d_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq, img_lq_d = padding_DP(img_gt, img_lq, img_lq_d, gt_size)

            # random crop
            img_gt, img_lq, img_lq_d = paired_random_crop_DP(img_gt, img_lq, img_lq_d, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq, img_lq_d = random_augmentation(img_gt, img_lq, img_lq_d)

        if self.opt['phase'] == 'val':
            gt_size = self.opt['crop_size'] if 'crop_size' in self.opt else None
            if gt_size is not None:  # or gt_size > 0:
                # padding
                if gt_size > 0:
                    img_gt, img_lq, img_lq_d = padding_DP(img_gt, img_lq, img_lq_d, gt_size)
                    img_gt, img_lq, img_lq_d = paired_random_crop_DP(img_gt, img_lq, img_lq_d, gt_size, scale,
                                                        gt_path)
            
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, img_lq_d = img2tensor([img_gt, img_lq, img_lq_d],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_lq_d, self.mean, self.std, inplace=True)
            
        # from basicsr.utils import imwrite, tensor2img
        # sr_img = tensor2img(img_lq, rgb2bgr=True)
        # sr_depth = tensor2img(img_lq_d, rgb2bgr=True)
        
        # imwrite(sr_img, '/data/dsq/Restormer/Restormer-main/experiments/ValSmall_DefocusDeblur_3DHistech_Restormer_Depth_SFT_GTNetE_gray1/visualization/img.png')
        # imwrite(sr_depth, '/data/dsq/Restormer/Restormer-main/experiments/ValSmall_DefocusDeblur_3DHistech_Restormer_Depth_SFT_GTNetE_gray1/visualization/depth.png')
        
        # print(lq_path)
        # print(lq_d_path)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_d': img_lq_d,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'lq_d_path': lq_d_path
        }

    def __len__(self):
        return len(self.paths)


    
class Dataset_PairedImage_depth3(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage_depth3, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_folder, self.lq_depth_folder, self.gt_depth_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_lq_depth'], opt['dataroot_gt_depth']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_Depth4_paths_from_folder(
                [self.lq_folder, self.gt_folder,self.lq_depth_folder, self.gt_depth_folder], ['lq', 'gt', 'lq_d', 'gt_d'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # index = 0
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))
        
        # print(gt_path)

        lq_path = self.paths[index]['lq_path']
        # print(lq_path)
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))
        
        lq_d_path = self.paths[index]['lq_d_path']
        img_bytes = self.file_client.get(lq_d_path, 'lq_d')
        try:
            img_lq_d = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_d_path))
        
        gt_d_path = self.paths[index]['gt_d_path']
        img_bytes = self.file_client.get(gt_d_path, 'gt_d')
        try:
            img_gt_d = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(gt_d_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq, img_lq_d, img_gt_d = padding_DP4(img_gt, img_lq, img_lq_d, img_gt_d, gt_size)

            # random crop
            img_gt, img_lq, img_lq_d, img_gt_d = paired_random_crop_DP4(img_gt, img_lq, img_lq_d, img_gt_d, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq, img_lq_d, img_gt_d = random_augmentation(img_gt, img_lq, img_lq_d, img_gt_d)

        if self.opt['phase'] == 'val':
            gt_size = self.opt['crop_size'] if 'crop_size' in self.opt else None
            if gt_size is not None:  # or gt_size > 0:
                # padding
                if gt_size > 0:
                    img_gt, img_lq, img_lq_d, img_gt_d = padding_DP4(img_gt, img_lq, img_lq_d, img_gt_d, gt_size)
                    img_gt, img_lq, img_lq_d, img_gt_d = paired_random_crop_DP4(img_gt, img_lq, img_lq_d, img_gt_d, gt_size, scale,
                                                        gt_path)
            
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, img_lq_d, img_gt_d = img2tensor([img_gt, img_lq, img_lq_d, img_gt_d],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_lq_d, self.mean, self.std, inplace=True)
            normalize(img_gt_d, self.mean, self.std, inplace=True)
            
        # from basicsr.utils import imwrite, tensor2img
        # sr_img = tensor2img(img_lq, rgb2bgr=True)
        # sr_depth = tensor2img(img_lq_d, rgb2bgr=True)
        
        # imwrite(sr_img, '/data/dsq/Restormer/Restormer-main/experiments/ValSmall_DefocusDeblur_3DHistech_Restormer_Depth_SFT_GTNetE_gray1/visualization/img.png')
        # imwrite(sr_depth, '/data/dsq/Restormer/Restormer-main/experiments/ValSmall_DefocusDeblur_3DHistech_Restormer_Depth_SFT_GTNetE_gray1/visualization/depth.png')
        
        # print(lq_path)
        # print(lq_d_path)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_d': img_lq_d,
            'gt_d': img_gt_d,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'lq_d_path': lq_d_path,
            'gt_d_path': gt_d_path
        }

    def __len__(self):
        return len(self.paths)
