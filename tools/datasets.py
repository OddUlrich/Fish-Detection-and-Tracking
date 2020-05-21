#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from skimage.transform import resize
#import skimage.transform

class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416):
        """
        Setup the path for images and labels in training dataset or validation dataset.
        The path from list_path should be rename to track label files. 
        Add "frame_" and '0's at the beginning and replace '.jpg' with '.txt' at the end.
        Bbox file format: data/obj_train_data/1.txt
        
        Input:
            list_path - /data/train.txt, or /data/val.txt. It contains image file directories.
                        format: data/obj_train_data/(number).jpg
        """
        
        with open(list_path, 'r') as fp:
            self.img_files = fp.readlines()
            
        # Rename files and set the new size for image.
        self.label_files = [path.replace('.jpg', '.txt') for path in self.img_files]
        self.img_size = img_size
        self.max_objects = 1
        
    def __getitem__(self, index):
        while True:
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            if not os.path.isfile(img_path):
                index += 1
                continue
            else:
                img = np.array(Image.open(img_path))
                if len(img.shape) != 3:
                    index += 1
                    continue
                else:
                    break
                
#        # Load the image and convert it into array.
#        img_path = self.img_files[index % len(self.img_files)].rstrip()
#        img = np.array(Image.open(img_path))
#        
#        # Skip images with less than three channels.
#        while len(img.shape) != 3:
#            index += 1
#            img_path = self.img_files[index % len(self.img_files)].rstrip()
#            img = np.array(Image.open(img_path))
    
        # Calculate the padding and add it to the image.
        h, w, _ = img.shape
        pix_diff = np.abs(h-w)
        pad1 = pix_diff // 2
        pad2 = pix_diff - pix_diff // 2
        padding = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    
        input_img = np.pad(img, padding, 'constant', constant_values=128) / 255
        padded_h, padded_w, _ = input_img.shape
    
        # Resize and normalize image, then convert to tensor.
        input_img = resize(input_img, (self.img_size, self.img_size, 3), mode='reflect')

        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = torch.from_numpy(input_img).float()
    
        
        # Label padding.
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        labels = None
        
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
        
            # Calculate the coordinate from image original info.
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)
    
            # Add padding and update labels.
            x1 += padding[1][0]
            x2 += padding[1][0]
            y1 += padding[0][0]
            y2 += padding[0][0]
            
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
            
        # Fill matrix    
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[: self.max_objects]] = labels[: self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)
        
        return img_path, input_img, filled_labels
    
    def __len__(self):
        return len(self.img_files)
    
    
    
    
    
    