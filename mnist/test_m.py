from __future__ import print_function
import os
if not os.path.isdir('mnist_m'):
    os.system('mkdir mnist_m/')
    os.system('rm -r mnist_m/')
    print("Downloading and setting up MNIST_M dataset")
    os.system('gdown --id 0B_tExHiYS-0veklUZHFYT19KYjg')
    os.system('tar -xf mnist_m.tar.gz')
else:
    pass

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from dataset import Mnist_m
from tqdm import tqdm
from model import classifier

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    counter=0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            test_loss += float(nn.CrossEntropyLoss()(output, target))  # sum up batch loss
            pred = (torch.nn.Softmax()(output)).argmax(dim=1, keepdim=True).squeeze()  # get the index of the max log-probability
            correct += float(torch.sum(pred==target))/len(target)
            counter+=1.0
            
            

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, counter,
        100. * correct / counter))


def main():

    use_cuda = True
    batch_size=64
    lr=0.1
    test_batch_size=128

    epochs=20
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32,32))
        ])

    dataset1 = Mnist_m('mnist_m',train=True,transform=transform)
    dataset2=Mnist_m('mnist_m',train=False,transform=transform)
    datasets=[dataset1,dataset2]
    dataset_test = ConcatDataset(datasets)
    
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    model = classifier().cuda()#.to(device)
    model.load_state_dict(torch.load('classifier_complete.pt'))

    test(model, device, test_loader)
    


if __name__ == '__main__':
    main()
