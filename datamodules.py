from torch.utils.data import Dataset, DataLoader
import os
import glob
from skimage import io
from PIL import Image
import numpy as np
import pytorch_lightning as pl
import torch
from constants import classes

class CAMELYON16_PATCHES(Dataset):
    def __init__(self, dir_src_normal, dir_src_tumor,
                 num_samples, transform=None
                 ):
        self.dir_src_normal = dir_src_normal
        self.num_samples = num_samples
        self.transform=transform
        path_normal = [os.path.join(dir_src_normal,'{}.png'.format(i)) for i in range(self.num_samples)]
        paths_tumor = [os.path.join(dir_src_tumor,'{}.png'.format(i)) for i in range(self.num_samples)]
        self.filepaths = path_normal+paths_tumor

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img

class Kather_19_patches(CAMELYON16_PATCHES):
    def __init__(self, dir_img, path_list,
                 transform=None, 
                 return_label = None,
                 ):
        self.dir_img = dir_img
        with open(path_list, 'r') as f:
            self.filepaths, self.labels = zip(*[(os.path.join(self.dir_img,line.strip()), line.strip().split(os.sep)[0]) for line in f.readlines()])

        self.transform=transform
        self.return_label = return_label
        self.mapper = {}
        for i,cls in enumerate(classes):
            self.mapper[cls] = np.zeros(len(classes))
            self.mapper[cls][i] = 1.0
        
    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx])
        if self.transform is not None:
            img = self.transform(img)
        if self.return_label == 'categorical':
            return img, self.labels[idx]
        if self.return_label == 'one_hot':
            return img, torch.tensor(self.mapper[self.labels[idx]]).float()
        if self.return_label == 'numerical':
            return img, torch.tensor(np.argmax(self.mapper[self.labels[idx]]))
        return img

class Kather_19_ssl_multiscale(Kather_19_patches):
    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx])
        img_multiscale = self.transform['ms'](img)
        scales = len(img_multiscale)
        img1 = torch.zeros(scales,3,224,224).float()
        img2 = torch.zeros(scales,3,224,224).float()
        for i,img in enumerate(img_multiscale):
            img1[i] = self.transform['per_image'](img)
            img2[i] = self.transform['per_image'](img)

        if self.return_label == 'categorical':
            return img1, self.labels[idx]
        if self.return_label == 'one_hot':
            return img1, torch.tensor(self.mapper[self.labels[idx]]).float()
        if self.return_label == 'numerical':
            return img1, torch.tensor(np.argmax(self.mapper[self.labels[idx]]))
        if self.return_label is None:
            return img1, img2

class Kather_19_ssl(Kather_19_patches):
    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx])
        img1 = self.transform(img)
        img2 = self.transform(img)
        return img1,img2




class CAMELYONDataModule(pl.LightningDataModule):
    def __init__(self, dataset_train, dataset_val, batch_size: int = 32,
                 num_workers: int = 16,
                 transform_train = None,
                 transform_valid = None,
                 name = None
                 ):
        super().__init__()
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform_train = transform_train
        self.transform_valid = transform_valid
        self.name = name

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last = True
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

class KatherDataset(torch.utils.data.Dataset):
    def __init__(self, paths, y, transforms):
        super().__init__()
        self.paths = paths
        self.y = y
        self.transforms = transforms
    def __len__(self):
        return(len(self.paths))
    def __getitem__(self,idx):
        img = Image.open(self.paths[idx])
        if self.transforms is not None:
            img = self.transforms(img)
        flname = os.path.basename(self.paths[idx])
        return(img, self.y[idx], flname)


if __name__ == '__main__':
    dataset = Kather_19_patches(
            '/tank/mirror/kather-19/NCT-CRC-HE-100K',
            '/tank/mirror/kather-19/path_list_train.txt')

    print(dataset[0].size)
