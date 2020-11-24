from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from model import classifier


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    log_interval=10
    loss_sum=0.0
    from tqdm.auto import tqdm
    for (data, target) in tqdm(train_loader):
        data, target = data.to(device), target.to(device)
        mask=(torch.min(data)+torch.max(data))/3.0
        data[data>=mask]=1
        data[data<mask]=0
        data=data.repeat(1,3,1,1)


        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        loss_sum+=float(loss)/(len(train_loader.dataset)//data.shape[0])
        optimizer.step()
        #if batch_idx % log_interval == 0:
    print('Epoch {} Train loss {}'.format(epoch, loss_sum))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data, target = data.to(device), target.to(device)
            mask=(torch.min(data)+torch.max(data))/3.0
            data[data>=mask]=1
            data[data<mask]=0
            data=data.repeat(1,3,1,1)

            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    torch.save(model.state_dict(), 'classifier_basic.pt')


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

    scheduler = StepLR(optimizer, step_size=1,gamma=gamma)
    for epoch in range(1, epochs + 1):
        print("Epoch {}".format(epoch))
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()



if __name__ == '__main__':
    main()
