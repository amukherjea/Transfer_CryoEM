import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F
import cv2
import matplotlib.pyplot as plt
from model import encode_mnist

img=cv2.resize(cv2.imread('checker.png',0),(32,32))

img[img!=255]=0
img[img==255]=1
orig=img.copy()
img=torch.tensor(img).repeat(1,1,1).cuda()
img=img.unsqueeze(1).float()
#print(torch.unique(img[0][0]))
img=encode_mnist().cuda()(img)[0]
#print(torch.unique(img[0,0,:,:].detach()))
#print(img[0,0,:,:])
#print(img.shape)
img=img.detach().cpu().squeeze().numpy().transpose(1,2,0)
#print(img.shape)
f, axarr = plt.subplots(1,2)
axarr[0].imshow(orig)
axarr[1].imshow(img)

plt.show()

