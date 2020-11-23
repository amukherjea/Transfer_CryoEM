import os
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class Mnist_m(Dataset):
    def __init__(self, mnist_folder,train=True,transform=None):
        assert os.path.exists(mnist_folder), 'Master Folder not exists'
        self.train=train
        self.transform=transform
        if train:
            self.mnistfolder = os.path.abspath(mnist_folder+'/mnist_m_train/')
            
        else:
            self.mnistfolder = os.path.abspath(mnist_folder+'/mnist_m_test/')
            
        self.labels_file = self.mnistfolder+'_labels.txt'
        
        self.labels = pd.read_csv(self.labels_file, sep=" ",header = None).values
        
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        file_=self.labels[index]
        img_filename = self.mnistfolder+'/'+file_[0]
        label=file_[1]
        
        img = Image.open(img_filename)

        if self.transform:
            img=self.transform(img)
        return img, label

if __name__ == '__main__':
    
    ds = Mnist_m('mnist_m',train=True)

    img, label = ds.__getitem__(10)
    print(img.shape,label)
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()
