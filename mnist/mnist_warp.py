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
import wandb #remove before push 
wandb.init(project="deceptionnet") #remove before push
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.classifier=classifier()
        self.encoder=encode_mnist()
    def forward(self,x):
        #print(x.shape)
        out,loss_bar,dec2=self.encoder(x)
        #print("BG",dec2['background'][0])
        #print("FG",dec2['foreground'][0])
        #out_preds=self.classifier(out)
        #loss_bar=0
        return out,loss_bar

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



def train(model,model2, device, train_loader, optimizer, epoch):
    model.eval()
    model2.train()
    log_interval=10
    loss_sum=0.0
    from tqdm.auto import tqdm
    for (data, target) in tqdm(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output,loss_weight = model2(data)
        #print(output[0][0])
        #print(torch.unique(output[0].detach()))
        cv2.imwrite('demo.png',cv2.pyrUp(cv2.pyrUp(output.detach().cpu().numpy()[0].transpose(1,2,0)*255)))
        cv2.imwrite('demo_orig.png',cv2.pyrUp(cv2.pyrUp(data.detach().cpu().numpy()[0].transpose(1,2,0)*255)))
        output_=model(output)
        #print(data.size())
        loss_sep=(loss_weight.mean())
        loss_class = (0-F.nll_loss(output_, target))*1e+2
        loss=loss_class+loss_sep
        loss.backward()

        loss_sum+=np.abs(float(loss)/(len(train_loader.dataset)//data.shape[0]))
        #print("Loss_Sep {} Loss_Classification {}".format((float(loss_sep)),np.abs(float(loss_class))))
        optimizer.step()
        #break
        #if batch_idx % log_interval == 0:
    wandb.log({"Train Loss: Warp": loss_sum}) #remove before push 
    print('Epoch {} Train loss {}'.format(epoch, loss_sum))


def test(model,model2, device, test_loader):
    model2.eval()
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output,loss_weight = model2(data)
            output=model(output)
            test_loss += np.abs((-F.nll_loss(output, target, reduction='sum').cpu().item())+(loss_weight.mean().cpu()))  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    wandb.log({"Test Accuracy:Warp": 100. * correct / len(test_loader.dataset), "Test Loss:Warp": test_loss}) #remove before push
    torch.save(model2.state_dict(), 'classifier_advanced.pt')


def main():
    # Training settings

    use_cuda = True
    gamma=0.7
    save_model=True
    batch_size=45 #128
    lr=0.1
    test_batch_size=128

    epochs=20
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 6,
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
    wandb.watch(model)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1,gamma=gamma)
    model2=Net().cuda()
    model.load_state_dict(torch.load('classifier_basic.pt'))
    # for param in model.parameters():
    #     param.requires_grad = False
    optimizer = optim.Adadelta(model2.parameters(), lr=0.01)

    for epoch in range(1, epochs + 1):
        print("Epoch {}".format(epoch))
        train(model, model2,device, train_loader, optimizer, epoch)
        test(model, model2,device, test_loader)
        scheduler.step()

    if save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main()
