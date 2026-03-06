from copy import deepcopy
import h5py
import math
import json
import numpy as np
from PIL import Image
import random
from scipy.ndimage.interpolation import zoom
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from dataset.transform import random_rot_flip, random_rotate, blur, obtain_cutmix_box
from scipy.ndimage import gaussian_filter, map_coordinates
import cv2


def elastic_transform(image, mask, alpha=30, sigma=5):
    """弹性形变，模拟不同扫描角度下的斑块形态"""
    random_state = np.random.RandomState(None)
    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    image_t = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    mask_t = map_coordinates(mask, indices, order=0, mode='reflect').reshape(shape)

    return image_t, mask_t


def adaptive_histogram_equalization(image):
    """CLAHE 对比度增强，突出斑块边界"""
    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image_uint8)
    return enhanced.astype(np.float32) / 255.0


class CSVSemiDataset(Dataset):
    def __init__(self, json_file_path, mode, size=None, n_sample=None):
        self.json_file_path = json_file_path
        self.mode = mode
        self.size = size
        self.n_sample = n_sample

        if mode == 'train_l' or mode == 'train_u':
            with open(self.json_file_path, mode='r') as f:
                self.case_list = json.load(f)
            if mode == 'train_l' and n_sample is not None:
                self.case_list *= math.ceil(n_sample / len(self.case_list))
                self.case_list = self.case_list[:n_sample]
        else:
            with open(self.json_file_path, mode='r') as f:
                self.case_list = json.load(f)

    def _read_pair(self, image_h5_file):
        with h5py.File(image_h5_file, 'r') as f:
            long_img = f['long_img'][:]
            trans_img = f['trans_img'][:]
        # ensure float32 and normalize to [0,1] (many h5 stores 0-255)
        long_img = long_img.astype(np.float32)
        trans_img = trans_img.astype(np.float32)
        # only scale if values appear to be in 0-255 range
        try:
            if long_img.max() > 1.0:
                long_img = long_img / 255.0
        except ValueError:
            # empty arrays or unexpected shapes: skip scaling
            pass
        try:
            if trans_img.max() > 1.0:
                trans_img = trans_img / 255.0
        except ValueError:
            pass
        return long_img, trans_img

    def _read_label(self, label_h5_file):
        with h5py.File(label_h5_file, 'r') as f:
            long_mask = f['long_mask'][:]
            trans_mask = f['trans_mask'][:]
            cls = f['cls'][()]   # shape: [] or [1] or [C]
        # map possible mask values {0,128,255} -> {0,1,2}
        long_mask = long_mask.astype(np.int64)
        trans_mask = trans_mask.astype(np.int64)
        # map 128 -> 1, 255 -> 2, leave 0 as 0 (robust if masks already 0/1/2)
        long_mask = np.where(long_mask == 128, 1, long_mask)
        long_mask = np.where(long_mask == 255, 2, long_mask)
        trans_mask = np.where(trans_mask == 128, 1, trans_mask)
        trans_mask = np.where(trans_mask == 255, 2, trans_mask)
        return long_mask, trans_mask, cls
    

    def __getitem__(self, item):
        case = self.case_list[item]

        if self.mode == 'valid':
            image_h5_file, label_h5_file = case['image'], case['label']
            long_img, trans_img = self._read_pair(image_h5_file)
            long_mask, trans_mask, cls = self._read_label(label_h5_file)

            return (torch.from_numpy(long_img).unsqueeze(0).float(),
                    torch.from_numpy(trans_img).unsqueeze(0).float(),
                    torch.from_numpy(long_mask).long(),
                    torch.from_numpy(trans_mask).long(),
                    torch.tensor(cls).long())

        elif self.mode == 'train_l':
            image_h5_file, label_h5_file = case['image'], case['label']
            long_img, trans_img = self._read_pair(image_h5_file)
            long_mask, trans_mask, cls = self._read_label(label_h5_file)

            # === 新增：对比度增强（50%概率） ===
            if random.random() > 0.5:
                long_img = adaptive_histogram_equalization(long_img)
            if random.random() > 0.5:
                trans_img = adaptive_histogram_equalization(trans_img)

            # === 新增：弹性形变（30%概率，针对斑块类） ===
            if random.random() > 0.7:
                long_img, long_mask = elastic_transform(long_img, long_mask, alpha=30, sigma=5)
            if random.random() > 0.7:
                trans_img, trans_mask = elastic_transform(trans_img, trans_mask, alpha=30, sigma=5)

            # Apply same-type augmentation to long/trans (each can be independently random)
            if random.random() > 0.5:
                long_img, long_mask = random_rot_flip(long_img, long_mask)
                trans_img, trans_mask = random_rot_flip(trans_img, trans_mask)
            elif random.random() > 0.5:
                long_img, long_mask = random_rotate(long_img, long_mask)
                trans_img, trans_mask = random_rotate(trans_img, trans_mask)

            # Resize to target size
            x, y = long_img.shape
            long_img = zoom(long_img, (self.size / x, self.size / y), order=0)
            long_mask = zoom(long_mask, (self.size / x, self.size / y), order=0)

            x2, y2 = trans_img.shape
            trans_img = zoom(trans_img, (self.size / x2, self.size / y2), order=0)
            trans_mask = zoom(trans_mask, (self.size / x2, self.size / y2), order=0)

            return (torch.from_numpy(long_img).unsqueeze(0).float(),
                    torch.from_numpy(trans_img).unsqueeze(0).float(),
                    torch.from_numpy(long_mask).long(),
                    torch.from_numpy(trans_mask).long(),
                    torch.tensor(cls).long())

        elif self.mode == 'train_u':
            image_h5_file = case['image']
            long_img, trans_img = self._read_pair(image_h5_file)

            def _make_u(img):
                # Matches previous logic; helper for producing weak/strong variants
                if random.random() > 0.5:
                    img = random_rot_flip(img)
                elif random.random() > 0.5:
                    img = random_rotate(img)

                x, y = img.shape
                img = zoom(img, (self.size / x, self.size / y), order=0)

                img = Image.fromarray((img * 255).astype(np.uint8))
                img_s1, img_s2 = deepcopy(img), deepcopy(img)
                img_w = torch.from_numpy(np.array(img)).unsqueeze(0).float() / 255.0

                if random.random() < 0.8:
                    img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
                img_s1 = blur(img_s1, p=0.5)
                box1 = obtain_cutmix_box(self.size, p=0.5)
                img_s1 = torch.from_numpy(np.array(img_s1)).unsqueeze(0).float() / 255.0

                if random.random() < 0.8:
                    img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
                img_s2 = blur(img_s2, p=0.5)
                box2 = obtain_cutmix_box(self.size, p=0.5)
                img_s2 = torch.from_numpy(np.array(img_s2)).unsqueeze(0).float() / 255.0

                return img_w, img_s1, img_s2, box1, box2

            long_w, long_s1, long_s2, box_l1, box_l2 = _make_u(long_img)
            trans_w, trans_s1, trans_s2, box_t1, box_t2 = _make_u(trans_img)

            return long_w, long_s1, long_s2, box_l1, box_l2, trans_w, trans_s1, trans_s2, box_t1, box_t2

    def __len__(self):
        return len(self.case_list)



