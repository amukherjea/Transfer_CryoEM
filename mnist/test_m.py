import os
try:
    os.system('mkdir mnist_m/')
    os.system('rm -r mnist_m/')
except:
    print("Downloading and setting up MNIST_M dataset")
    os.system('gdown --id 0B_tExHiYS-0veklUZHFYT19KYjg')
    os.system('tar -xf mnist_m.tar.gz')
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from dataset import Mnist_m
from tqdm.auto import tqdm
class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        #print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


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
    model.load_state_dict(torch.load('classifier_basic.pt'))

    test(model, device, test_loader)
    


if __name__ == '__main__':
    main()
