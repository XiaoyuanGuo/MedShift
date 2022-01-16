import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset

   
input_size = 224   
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.1892,], [0.0888,])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.1892,], [0.0888,])
    ]),
}

class MURA_Allclass_Dataset(Dataset):
    def __init__(self, data, mode='train'):
        self.data = data
        self.mode = mode   
        if self.mode == "train":
            self.transformations = data_transforms["train"]
        else:
            self.transformations = data_transforms["val"]
         
        
    def __getitem__(self, idx):
        path, label = self.data[idx][0], self.data[idx][1]
        if self.mode == 'train' or self.mode == 'val':
            img = Image.open(path) # all to grayscale
            t = (np.array(img)/np.array(img).max())*255.0
            img = Image.fromarray(t).convert("L")
        else:
            img = Image.open(path).convert("L")
            
        img = img.resize((imgSize,imgSize)) # resize to a uniform size
        if self.transformations != None:
            img = self.transformations(img)
        image = torch.stack([img,img,img], axis = 1)
        img = torch.squeeze(image, dim = 0).type(torch.FloatTensor)
        if self.mode == 'train' or self.mode == "val":
            return img, label
        else:
            return img, path, label
      
    def __len__(self): 
        return len(self.data)
    
    

    
