from audioop import reverse
from distutils import core
from email.mime import base
from random import random
from webbrowser import Elinks
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

import os
import pdb
import cv2
import random
import nibabel as nib
import torchvision.transforms as transforms 
from PIL import Image 


def data_collate(batch):
    input=None
    target = None
    #input_paths = None
    total_num =0
    #num_per_patient = []
    for info in batch:
      if total_num==0:
        input = torch.from_numpy(info[0]).unsqueeze(0)
        target = torch.from_numpy(info[1]).unsqueeze(0)
        #input_paths = info[3]
      else:
        input = torch.cat((input, torch.from_numpy(info[0]).unsqueeze(0)))
        target = torch.cat((target, torch.from_numpy(info[1]).unsqueeze(0)))
        #input_paths = np.dstack((input_paths, info[3]))
      #num_per_patient.append(info[2])
      total_num+=1

    return input.float(), target #,  num_per_patient, input_paths, info[4]

def get_dataset_path(dataset='pancreas'):
    files = ['train_lab.txt', 'train_unlab.txt', 'test.txt']
    return ['/'.join(['data_lists', dataset, f]) for f in files]

def trans_npy2nii(image, label, data_idx, aug='0'):
    to_path = 'result/visiable/kits19/'
    image = np.array(image.squeeze())
    label = np.array(label)
    new_image = nib.Nifti1Image(np.array(image), np.eye(4))
    new_label = nib.Nifti1Image(np.array(label), np.eye(4))
    nib.save(new_image, to_path+str(data_idx)+f'_aug{aug}_img.nii.gz')
    if aug == '0':
        nib.save(new_label, to_path+str(data_idx)+f'_label.nii.gz')
    print("Save {} file h52nii_trans".format(str(data_idx)))


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def _get_transform(self, label):
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 1, 0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = label.shape
        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        def do_transform(x):
            if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
                x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            x = x[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return x

        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def _get_transform(self, x):
        if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - x.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - x.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - x.shape[2]) // 2 + 1, 0)
            x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = x.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        def do_transform(image):
            if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= self.output_size[2]:
                try:
                    image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                except Exception as e:
                    print(e)
            image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return image

        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]

def _get_cordi(x):
    (w, h, d) = x.shape
    w1 = np.random.randint(0, w - 128)
    h1 = np.random.randint(0, h - 128)
    d1 = np.random.randint(0, d - 128)
    return w1, h1, d1

class RandomROICrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def _get_transform(self, x):
        if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - x.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - x.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - x.shape[2]) // 2 + 1, 0)
            x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        w1, h1, d1 = _get_cordi(x)
        x1 = x[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]].sum()
        x2 = x.sum() / 5
        while x1 < x2 :
            w1, h1, d1 = _get_cordi(x)
            x1 = x[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]].sum()
            x2 = x.sum() / 5

        def do_transform(image):
            if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= self.output_size[2]:
                try:
                    image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                except Exception as e:
                    print(e)
            image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return image

        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[1])
        return [transform(s) for s in samples]


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample[0]
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        sample = [image] + [*sample[1:]]
        return [torch.from_numpy(s.astype(np.float32)) for s in sample]


class Pancreas(Dataset):
    """ Pancreas Dataset """
    def __init__(self, base_dir, name, split, no_crop=False, reverse=False, TTA=False):
        self._base_dir = base_dir
        self.split = split
        self.reverse=reverse

        tr_transform = Compose([
            # RandomRotFlip(),
            RandomCrop((96, 96, 96)),
            # RandomNoise(),
            ToTensor()
        ])
        if no_crop:
            test_transform = Compose([
                # CenterCrop((160, 160, 128)),
                CenterCrop((96, 96, 96)),
                ToTensor()
            ])
        else:
            test_transform = Compose([
                CenterCrop((96, 96, 96)),
                ToTensor()
            ])

        data_list_paths = get_dataset_path(name)

        if split == 'train_lab':
            data_path = data_list_paths[0]
            self.transform = tr_transform
        elif split == 'train_unlab':
            data_path = data_list_paths[1]
            self.transform = test_transform  # tr_transform
        else:
            data_path = data_list_paths[2]
            self.transform = test_transform

        with open(data_path, 'r') as f:
            self.image_list = f.readlines()

        self.image_list = [self._base_dir + "/{}".format(item.strip()) for item in self.image_list]
        print("Split : {}, total {} samples".format(split, len(self.image_list)))

    def __len__(self):
        if self.split == 'train_lab':
            return len(self.image_list) * 5
        else:
            return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx % len(self.image_list)]
        if self.reverse:
            image_path = self.image_list[len(self.image_list) - idx % len(self.image_list) - 1]
        h5f = h5py.File(image_path, 'r')
        image, label = h5f['image'][:], h5f['label'][:].astype(np.float32)
        samples = image, label
        if self.transform:
            tr_samples = self.transform(samples)
        image_, label_ = tr_samples
        return image_.float(), label_.long()


class cutmix_Pancreas(Dataset):
    """ Pancreas Dataset """

    def __init__(self, base_dir, name, split, no_crop=False, TTA=False, patch_dim=32):
        self._base_dir = base_dir
        self.split = split
        dim = patch_dim
        tr_transform = Compose([
            # RandomRotFlip(),
            RandomCrop((dim, dim, dim)),
            # RandomNoise(),
            ToTensor()
        ])
        if no_crop:
            test_transform = Compose([
                # CenterCrop((160, 160, 128)),
                CenterCrop((dim, dim, dim)),
                ToTensor()
            ])
        else:
            test_transform = Compose([
                CenterCrop((dim, dim, dim)),
                ToTensor()
            ])

        data_list_paths = get_dataset_path(name)

        if split == 'train_lab':
            data_path = data_list_paths[0]
            self.transform = tr_transform
        elif split == 'train_unlab':
            data_path = data_list_paths[1]
            self.transform = test_transform  # tr_transform
        else:
            data_path = data_list_paths[2]
            self.transform = test_transform

        with open(data_path, 'r') as f:
            self.image_list = f.readlines()

        self.image_list = [self._base_dir + "/{}".format(item.strip()) for item in self.image_list]
        print("Split : {}, total {} samples".format(split, len(self.image_list)))

    def __len__(self):
        if self.split == 'train_lab':
            return len(self.image_list) * 5
        else:
            return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[len(self.image_list) - idx % len(self.image_list) - 1]
        h5f = h5py.File(image_path, 'r')
        image, label = h5f['image'][:], h5f['label'][:].astype(np.float32)
        samples = image, label
        if self.transform:
            tr_samples = self.transform(samples)
        image_, label_ = tr_samples
        return image_.float(), label_.long()



class PancreasSTDataset(Dataset):
    def __init__(self, imgs, plabs, masks, labs):
        self.img = [img.cpu().squeeze().numpy() for img in imgs]
        self.plab = [np.squeeze(lab.cpu().numpy()) for lab in plabs]
        self.mask = [np.squeeze(mask.cpu().numpy()) for mask in masks]
        self.lab = [np.squeeze(lab.cpu().numpy()) for lab in labs]
        self.num = len(self.img)
        self.tr_transform = Compose([
            CenterCrop((96, 96, 96)),
            ToTensor()
        ])

    def __getitem__(self, idx):
        samples = self.img[idx], self.plab[idx], self.mask[idx], self.lab[idx]
        samples = self.tr_transform(samples)
        imgs, plabs, masks, labs = samples
        return imgs, plabs.long(), masks.float(), labs.long()

    def __len__(self):
        return self.num


# class Kits19(Dataset):
#     """KiTS19 dataset"""
#     def __init__(self, base_dir, split, labelnum=16, reverse=False, augmentation=None):
#         self.base_dir = base_dir
#         self.split = split
#         self.reverse = reverse
#         self.labelnum = labelnum
#         self.augmentation = augmentation

#         tr_transform = Compose([
#             RandomCrop((128, 128, 128)),
#             ToTensor()
#         ])
#         test_transform = Compose([
#             CenterCrop((128, 128, 128)),
#             ToTensor()
#         ])
#         if split == 'train_lab':
#             data_idx = range(0, labelnum)
#             #data_idx = range(160 - labelnum, 160)
#             self.transform = tr_transform
#         elif split == 'train_unlab':
#             data_idx = range(labelnum, 160)
#             #data_idx = range(0, 160 - labelnum)
#             self.transform = test_transform
#         else:
#             data_idx = range(160, 210)
#             self.transform = test_transform
#         self.data_idlist = data_idx
#         self.image_list = [self.base_dir + f"/case_{case_id:05}/volume.h5" for case_id in self.data_idlist]
#         print("Split : {}, total {} samples".format(split, len(self.data_idlist)))

#     def __len__(self):
#         if self.split == 'train_lab' and self.labelnum == 16:
#             return len(self.data_idlist) * 9
#         elif self.split == 'train_lab' and self.labelnum == 4:
#             return len(self.data_idlist) * 39
#         else:
#             return len(self.data_idlist)

#     def __getitem__(self, idx):
#         idx = idx % len(self.data_idlist)
#         if self.reverse:
#             idx = len(self.data_idlist) - idx - 1
#         image_path = os.path.join(self.base_dir, f'case_{self.data_idlist[idx]:05}/volume.h5')
#         h5f = h5py.File(image_path, 'r')
#         image, label = h5f['image'][:], h5f['label'][:].astype(np.float32)
#         samples = image, label
#         if self.transform:
#             tr_samples = self.transform(samples)
#         image_, label_ = tr_samples
#         #trans_npy2nii(image_, label_, data_idx=idx, aug='0')
#         if self.augmentation:
#             image_ = self.augmentation(image_)
#             #trans_npy2nii(image_, label_, data_idx=idx, aug='1')
#         return image_.float(), label_.long()

if __name__ == '__main__':
    pass
