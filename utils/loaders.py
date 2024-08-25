import pytorch_lightning as pl

import torch
import os
import PIL
import random
from PIL import Image
from typing import Any, Callable, List, Optional, Union, Tuple
import numpy as np
import cv2
from dataclasses import dataclass
import json
import pydicom
from torchvision import datasets
from torch.utils.data import Subset
import torchvision

def load_txt(path :str) -> list:
    return [line.rstrip('\n') for line in open(path)]


def check_channels(img):
    b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    if (b==g).all() and (b==r).all(): return True
    return False

class Img_To_1(object):
    def __call__(self, img):
        return img/255.0
    

class MNISTLoader(torch.utils.data.Dataset):


    def __init__(self, img_dir, transform = None):

        self.img_dir = img_dir
        self.transform = transform
        self.filenames = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.img_dir, self.filenames[idx]))
        patches = self.split_image(image)
 
        if self.transform:
            patches = self.transform(patches)
        return patches
   

class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root :str, name :str,
                 transform=None, target_transform=None, severity=1):
 
        super(CIFAR10C, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')
        
        self.data = np.load(data_path)[(severity-1)*10000:severity*10000]
        self.targets = np.load(target_path)[(severity-1)*10000:severity*10000]

        
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img, targets
    
    def __len__(self):
        return len(self.data)


def extract_subset(dataset, num_subset :int, random_subset :bool):
    if random_subset:
        random.seed(0)
        indices = random.sample(list(range(len(dataset))), num_subset)
    else:
        indices = [i for i in range(num_subset)]
    return Subset(dataset, indices)

class TinyImageNet(torchvision.datasets.VisionDataset):
    # load all images in train/val/test
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(TinyImageNet, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split
        self.data = []
        self.targets = []
        self.classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.load_data()

    def load_data(self):

        if self.split == 'train':
            data_path = os.path.join(self.root, 'train')
            print("loading train data")
            # find all the images in the train directory
            for class_name in os.listdir(data_path):
                class_path = os.path.join(data_path, class_name, 'images')
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    # read image as rgb
                    img = np.array(Image.open(img_path).convert('RGB'))
                    # if not rgb, convert to rgb
                    # if len(img.shape) == 2:
                    #     img = np.stack((img,)*3, axis=-1)
                    self.data.append(img)
                    self.targets.append(class_name)

        elif self.split == 'val':
            data_path = os.path.join(self.root, 'val')
            print("loading val data")
            with open(os.path.join(data_path, 'val_annotations.txt')) as f:
                for line in f:
                    img_name, class_name = line.split('\t')[:2]
                    img_path = os.path.join(data_path, 'images', img_name)
                    # temp open image

                    img = np.array(Image.open(img_path))
                    if len(img.shape) == 2:
                        img = np.stack((img,)*3, axis=-1)
                    self.data.append(img)
                    self.targets.append(class_name)

        elif self.split == 'test':
            data_path = os.path.join(self.root, 'test', 'images')
            print("loading test data")
            for img_name in os.listdir(data_path):
                img_path = os.path.join(data_path, img_name)
                img = np.array(Image.open(img_path))
                if len(img.shape) == 2:
                        img = np.stack((img,)*3, axis=-1)
                self.data.append(img)
                self.targets.append('')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img, targets
    


class TinyImageNetC(torchvision.datasets.VisionDataset):
    def __init__(self, root :str, curruption :str, transform=None, target_transform=None, severity=1):
        super(TinyImageNetC, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(root, curruption, str(severity))
        target_path = os.listdir(os.path.join(root, curruption, str(severity)))

        self.data = [] 
        self.targets = []

        self.__load_data__(data_path)

                
    
    def __len__(self):
        return len(self.data)
    
    def __load_data__(self, path):
        for class_name in os.listdir(path):
            for img_name in os.listdir(os.path.join(path, class_name)):
                img_path = os.path.join(path, class_name, img_name)
                img = np.array(Image.open(img_path))
                if len(img.shape) == 2:
                    img = np.stack((img,)*3, axis=-1)
                self.data.append(img)
                self.targets.append(class_name)

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img, targets
    

    
class ImageOOD(torchvision.datasets.VisionDataset):
    def __init__(self, root, transform=None, target_transform=None, data_to_load='OpenImageOOD.txt'):
        super(ImageOOD, self).__init__(root, transform=transform, target_transform=target_transform)
        self.data = []
        self.targets = []
        self.file_list = []

        # read the file list
        with open(data_to_load, 'r') as f:
            self.file_list = f.readlines()

        self.load_data()

    def load_data(self):
        data_path = self.root # os.path.join(self.root, 'data', 'ssb_hard')

        for line in self.file_list:

            img_path, _ = line.split()
            target = img_path.split('/')[1]

            img_path = os.path.join(data_path, img_path)
            img = np.array(Image.open(img_path).convert(mode='RGB'))
                
            self.data.append(img)
            self.targets.append(target)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img, targets
    