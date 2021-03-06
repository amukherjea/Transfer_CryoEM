from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
from model import classifier
import numpy as np
import cv2
from model import encode_mnist

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.encoder=encode_mnist()
    def forward(self,x):
        out,loss_bar,dec2=self.encoder(x)
        return out,loss_bar

def create_dataset(device,data,target,model2):
    with torch.no_grad():
            data, target = data.to(device), target.to(device)
            output,loss_weight = model2(data)
            mask=(torch.min(data)+torch.max(data))/3.0
            data[data>=mask]=1
            data[data<mask]=0
            data=data.repeat(1,3,1,1)
            data[:,0,:,:]=data[:,0,:,:]*np.random.random(1)[0]
            data[:,1,:,:]=data[:,1,:,:]*np.random.random(1)[0]
            data[:,2,:,:]=data[:,2,:,:]*np.random.random(1)[0]
            data[:,0,:,:][data[:,0,:,:]==0]=np.random.random(1)[0]
            data[:,1,:,:][data[:,1,:,:]==0]=np.random.random(1)[0]
            data[:,2,:,:][data[:,2,:,:]==0]=np.random.random(1)[0]

            #print(output[0][0])
            #print(torch.unique(output[0].detach()))
            # cv2.imwrite('demo.png',cv2.pyrUp(cv2.pyrUp(output.detach().cpu().numpy()[0].transpose(1,2,0)*255)))
            cv2.imwrite('demo_orig.png',cv2.pyrUp(cv2.pyrUp(data.detach().cpu().numpy()[0].transpose(1,2,0)*255)))
            
            output__=torch.cat([data,output],dim=0)
            target_=torch.cat([target,target],dim=0)
            #print(target.shape,target_.shape)
            return output__,target_#output__,target_#data.to(device),target.to(device)#
    

import os    
def main():
    # Training settings
    #os.system('python3 test_m.py')
    use_cuda = True
    gamma=0.7
    save_model=True
    batch_size=64
    lr=0.1
    test_batch_size=128
    epochs=50


    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 3,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32,32)),
        #transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = classifier().cuda()#.to(device)
#    model.load_state_dict(torch.load('classifier_basic.pt'))
    optimizer = optim.Adam(model.parameters(), lr=0.001,betas=(0.9,0.999))

    model2=Net().cuda().eval()
    model2.load_state_dict(torch.load('classifier_advanced.pt'))
    scheduler = ReduceLROnPlateau(optimizer,'min',factor=0.9,patience=2)
    #scheduler = StepLR(optimizer, step_size=1,gamma=gamma)
    for epoch in range(1, epochs + 1):
        print("Epoch {}".format(epoch))
        model.train()
        model2.eval()
        log_interval=10
        loss_sum=0.0
        from tqdm import tqdm
        for (data, target) in tqdm(train_loader):
            data, target =create_dataset(device,data,target,model2)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()

            loss_sum+=float(loss)/(len(train_loader.dataset)//data.shape[0])
            optimizer.step()
            #if batch_idx % log_interval == 0:
        print('Epoch {} Train loss {}'.format(epoch, loss_sum))
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target =create_dataset(device,data,target,model2)
                

                output = model(data)
                test_loss += nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, 2*len(test_loader.dataset),
            100. * correct / (2.0*len(test_loader.dataset))))
        
        torch.save(model.state_dict(), 'classifier_complete.pt')
        os.system('python3 test_m.py')
        scheduler.step(test_loss)



if __name__ == '__main__':
    main()
