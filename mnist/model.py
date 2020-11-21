import torch
from torch import nn
from torch.nn import functional as F
import numbers
import math
bottle=16
import numpy as np

class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_block, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        #print('conv',x.size())
        return self.net(x)

kernel_=3
padding_=1
thresh=1e-7

d = torch.linspace(-1, 1, 256)
meshx, meshy = torch.meshgrid((d, d))
grid = torch.stack((meshy, meshx), 2)


class decoder_ds(nn.Module):
    def __init__(self,in_channels=128):
        super(decoder_ds,self).__init__()
        global bottle
        self.c1=nn.Conv2d(in_channels=in_channels,out_channels=1,kernel_size=1,stride=1)
        self.f1=nn.Sequential(nn.Linear(bottle,1),nn.Hardtanh(0.1,5))

    def forward(self,x,orig):
        alpha=torch.flatten(self.c1(x),start_dim=1)
        #print(alpha.shape)
        alpha=self.f1(alpha)
        def_=deform(orig.shape).cuda()(x,orig,alpha)
        def_=def_.repeat(1,3,1,1)
        return def_




class deform(nn.Module):

    def __init__(self, shape,channels=1, kernel_size=5, sigma=4, stride=1, padding=2,dim=2):
        super(deform, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        k=torch.rand(*shape)
        k=k[:,0,:,:].unsqueeze(1)
        shape=k.shape
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        kernel = kernel.cuda() / torch.sum(kernel).cuda()

        kernel = kernel.view(1, 1, *kernel.size()).cuda()
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1)).cuda()

        #self.register_buffer('weight', kernel)
        self.weight=kernel
        self.groups = channels
        self.conv=F.conv2d
        self.padding=padding
        self.stride=stride
        self.shape=shape
        self.d1 = torch.linspace(0, shape[2]-1, shape[2]).cuda()
        self.d2 = torch.linspace(0, shape[3]-1, shape[3]).cuda()
        self.x, self.y = torch.meshgrid((self.d1, self.d2))

        self.x=self.x.cuda()
        self.x.requires_grad=False
        self.y=self.y.cuda()
        self.y.requires_grad=False


        self.dec=nn.Sequential(decoder_general(),nn.Hardtanh(-1,1))


    def forward(self, bottleneck, inputs,alphas):

        self.x=self.x.repeat(inputs.shape[0],1,1,1)
        self.y=self.y.repeat(inputs.shape[0],1,1,1)
        #print(self.x.shape)
        d=self.dec(bottleneck)

        dx=d[:,0,:,:].clone().unsqueeze(1)
        dy=d[:,1,:,:].clone().unsqueeze(1)
        dx=self.conv(dx.cuda(), weight=self.weight, groups=self.groups,stride=self.stride,padding=self.padding)
        dy=self.conv(dy.cuda(), weight=self.weight, groups=self.groups,stride=self.stride,padding=self.padding)
        for i in range(len(alphas)):
            dx[i]*=alphas[i]
            dy[i]*=alphas[i]

        #print(self.x.shape)
        x=(((self.x+dx)/(self.x.shape[2]-1)-0.5)*2).squeeze(1)
        y=(((self.y+dy)/(self.y.shape[3]-1)-0.5)*2).squeeze(1)
        x,y=y,x

        grid=torch.cat([x.unsqueeze(3),y.unsqueeze(3)],3)

        return torch.nn.functional.grid_sample(inputs,grid,align_corners=False)



class decoder_bg(nn.Module):
    def __init__(self):
        global bottle
        super(decoder_bg,self).__init__()
        in_channels=32
        out_channels=1

        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1)
        self.l1=nn.Sequential(nn.Linear(bottle,3),nn.Hardtanh(0,1))

        self.conv2=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1)
        self.l2=nn.Sequential(nn.Linear(bottle,3),nn.Hardtanh(0.1,0.9))


    def forward(self,x,orig):
        global thresh
        ll=x.shape[1]//2
        x1=x[:,:ll,:,:]
        x=x[:,ll:,:,:]
        s=(torch.flatten(self.conv1(x),start_dim=1))
        s=self.l1(s)
        s_for=self.l2(torch.flatten(self.conv2(x1),start_dim=1))
        mask=torch.abs(s-s_for)
        mask1=mask.clone()
        mask[mask1>thresh]=0
        mask[mask1<=thresh]=1
        #mask[mask>thresh],mask[mask<=thresh]=0,1
        #print(mask[0],s[0],s_for[0])

        diff_for_loss=(1/torch.abs(s-s_for))*mask

        col=orig
        #col=orig.clone()
        #col=col.repeat(1,3,1,1)
        for i in range(orig.shape[0]):
            for k in range(3):
                col[i,k,:,:][col[i,k,:,:]==0]=s[i,k]
                col[i,k,:,:][col[i,k,:,:]==1]=s_for[i,k]
                #print(s[i,k],s_for[i,k])
        #print(torch.unique(col[0][0].detach()))
        #print(diff_for_loss)
        return {'background':s,'foreground':s_for,'loss':(1e+3*diff_for_loss),'dec':col}

class decoder_ns(nn.Module):
    def __init__(self):
        super(decoder_ns,self).__init__()
        self.dec=nn.Sequential(decoder_general(in_channels=64,out_channels=3),nn.Hardtanh(-0.01,0.01))

    def forward(self,x):
        return self.dec(x)




class decoder_general(nn.Module):
    def __init__(self,in_channels=128,out_channels=2):
        super(decoder_general,self).__init__()
        # self.u1=nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        # self.u1_ = Conv_block(512,512)
        # self.u2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        # self.u2_ = Conv_block(256, 256)
        self.u3 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=2, stride=2)
        self.u3_= Conv_block(in_channels//2, in_channels//2)
        self.u4 = nn.ConvTranspose2d(in_channels=in_channels//2, out_channels=in_channels//4, kernel_size=2, stride=2)
        self.u4_= Conv_block(in_channels//4, in_channels//4)


        self.u5 = nn.ConvTranspose2d(in_channels=in_channels//4, out_channels=in_channels//8, kernel_size=2, stride=2)
        self.u5_=  nn.Sequential(
            nn.Conv2d(in_channels=in_channels//8, out_channels=in_channels//8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels//8, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )
    def forward(self,x):
        # x = self.u1(x)
        # x = self.u1_(x)
        # x = self.u2(x)
        # x = self.u2_(x)
        x = self.u3(x)
        x = self.u3_(x)
        x = self.u4(x)
        x = self.u4_(x)
        x = self.u5(x)
        x = self.u5_(x)
        #print(x.shape)
        return x
class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        global padding_,kernel_

        self.d0=Conv_block(1, 32)
        self.d1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(32, 64),
        )

        self.d2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(64, 128),
        )
        self.d3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(128, 256),
        )
        # self.d4 = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     Conv_block(256, 512),
        # )
        # self.d5 = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     Conv_block(512, 1024),
        # )


    def forward(self,x):
        #print('encoder',x.size())
        x_lose=self.d0(x)
        x = self.d1(x_lose)
        x = self.d2(x)
        x = self.d3(x)
        #x = self.d4(x)

        #x = self.d5(x)

        return x

class encode_mnist(nn.Module):
    def __init__(self):
        super(encode_mnist,self).__init__()
        self.encoder=encoder()
        self.decoder_1=decoder_ds()
        self.decoder_2=decoder_bg()
        self.decoder_3=decoder_ns()

    def forward(self,x):
        #print(x.size())
        x_en=self.encoder(x)
        dec1=self.decoder_1(x_en[:,:x_en.shape[1]//2,:,:],x)
        thresh=torch.min(dec1)+torch.max(dec1)/4.0
        dec1[dec1>thresh],dec1[dec1<=thresh]=1,0
        #print(torch.unique(dec1[0][0].detach()))

        dec2=self.decoder_2(x_en[:,((x_en.shape[1]*3)//4):,:,:],dec1)
        dec3=self.decoder_3(x_en[:,(x_en.shape[1]//2):((x_en.shape[1]*3)//4),:,:])
        out_avg=(dec2['dec']+dec3)

        return out_avg,dec2['loss'],dec2



in_=torch.rand(64,1,32,32).cuda()
in_[in_>0.5]=1
in_[in_<=0.5]=0
encoded=encode_mnist().cuda()(in_)

