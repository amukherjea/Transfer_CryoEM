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
        self.encoder=encode_mnist()
    def forward(self,x):
        out,loss_bar,dec2=self.encoder(x)
        return out,loss_bar




from torch.autograd import Function
class GradReverse(Function):
    @staticmethod
    def forward(self, x):
        return x.view_as(x)
    @staticmethod
    def backward(self, grad_output):
        return grad_output.neg()

def reverse_grad(x):
    return GradReverse.apply(x)

    

def main():
    # Training settings

    use_cuda = True
    gamma=0.7
    save_model=True
    batch_size=128 #128
    lr=0.01
    test_batch_size=256

    epochs=500
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 8,
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
    model2=Net().cuda()
    model.load_state_dict(torch.load('classifier_basic.pt'))
    optimizer = optim.Adam(model2.parameters(), lr=0.001,betas=(0.9,0.999))
    #optimizer = optim.SGD(model2.parameters(),lr=0.001,momentum=0.9,nesterov=True)
    #scheduler = StepLR(optimizer, step_size=1,gamma=gamma)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=3)
    for epoch in range(1, epochs + 1):
        print("Epoch {}".format(epoch))
        model2.train()
        log_interval=10
        loss_sum=0.0
        from tqdm import tqdm
        for (data, target) in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output,loss_weight = (model2((data)))
            #print(output[0][0])
            #print(torch.unique(output[0].detach()))
            cv2.imwrite('demo.png',cv2.pyrUp(cv2.pyrUp(output.detach().cpu().numpy()[0].transpose(1,2,0)*255)))
            cv2.imwrite('demo_orig.png',cv2.pyrUp(cv2.pyrUp(data.detach().cpu().numpy()[0].transpose(1,2,0)*255)))
            output_=(model(output))
            #print(data.size())
            loss_sep=(loss_weight.mean())
            loss_class = -(nn.CrossEntropyLoss()(output_, target))*1e+2
            loss=loss_class#+loss_sep
            loss_class.backward()
            
            #print("Loss Train",float(np.abs(float(loss))))

            loss_sum+=np.abs(float(loss)/(len(train_loader.dataset)//data.shape[0]))
            #print("Loss_Sep {} Loss_Classification {}".format((float(loss_sep)),np.abs(float(loss_class))))
            optimizer.step()
            #break
            #if batch_idx % log_interval == 0:
        print('Epoch {} Train loss {}'.format(epoch, loss_sum))

        model2.eval()
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output,loss_weight = model2(data)
                output=model(output)
                test_loss += np.abs((nn.CrossEntropyLoss()(output, target).cpu().item())+(loss_weight.mean().cpu()))  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        torch.save(model2.state_dict(), 'classifier_advanced.pt')


        scheduler.step(test_loss)
    

    if save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main()
