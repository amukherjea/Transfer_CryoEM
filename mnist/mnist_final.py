from __future__ import print_function
import argparse
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from model import encode_mnist
from model import classifier
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.classifier=classifier()
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

            #print(output[0][0])
            #print(torch.unique(output[0].detach()))
            # cv2.imwrite('demo.png',cv2.pyrUp(cv2.pyrUp(output.detach().cpu().numpy()[0].transpose(1,2,0)*255)))
            cv2.imwrite('demo_orig.png',cv2.pyrUp(cv2.pyrUp(data.detach().cpu().numpy()[0].transpose(1,2,0)*255)))
            
            output__=torch.cat([data,output],dim=0)
            target_=torch.cat([target,target],dim=0)
            #print(target.shape,target_.shape)
            return output__,target_#data.to(device),target.to(device)#
    

    


    
    
def main():
    # Training settings

    use_cuda = True
    gamma=0.7
    save_model=True
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
        transforms.Resize((32,32)),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = classifier().cuda()#.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01,betas=(0.9,0.999))
    model2=Net().cuda().eval()
    model2.load_state_dict(torch.load('classifier_advanced.pt'))
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.9,patience=3)
    for epoch in range(1, epochs + 1):
        print("Epoch {}".format(epoch))
        model.train()
        log_interval=10
        loss_sum=0.0
        from tqdm.auto import tqdm
        for (data, target) in tqdm(train_loader):
            out,tar=create_dataset(device,data,target,model2)
            
            optimizer.zero_grad()
            output=model(out)
            loss_class = (F.nll_loss(output, tar))*1e+2
            loss=loss_class
            loss.backward()

            loss_sum+=np.abs(float(loss)/(len(train_loader.dataset)//data.shape[0]))
            optimizer.step()
            
        print('Epoch {} Train loss {}'.format(epoch, loss_sum))

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                
                out,tar=create_dataset(device,data,target,model2)
                output=model(out)
        
                test_loss += np.abs((F.nll_loss(output, tar, reduction='sum').cpu().item()))  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(tar.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, (2*len(test_loader.dataset)),
            100. * correct / (2*len(test_loader.dataset))))
        torch.save(model.state_dict(), 'classifier_complete.pt')
       
        scheduler.step(test_loss)


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main()
