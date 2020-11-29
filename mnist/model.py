import torch
from torch import nn
from torch.nn import functional as F
import numbers
import math
bottle=16
import numpy as np

class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1,1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 48, 5, 1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2))
        self.fc=nn.Sequential(
            nn.Linear(192,100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100,10))
        

    def forward(self, x):
        
        x = self.layers(x)
        x = torch.flatten(x, 1)
        
        x=self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_block, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.net(x)

kernel_=3
padding_=1
thresh=1e-5

d = torch.linspace(-1, 1, 256)
meshx, meshy = torch.meshgrid((d, d))
grid = torch.stack((meshy, meshx), 2)


class decoder_ds(nn.Module):
    def __init__(self,in_channels=128):
        super(decoder_ds,self).__init__()
        global bottle
        self.c1=nn.Conv2d(in_channels=in_channels,out_channels=1,kernel_size=1,stride=1)
        self.f1=nn.Sequential(nn.Linear(bottle,1),nn.Sigmoid())

    def forward(self,x,orig):

        alpha=torch.flatten(self.c1(x['encode_ds']),start_dim=1)
        alpha=self.f1(alpha)*5.0
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

        channel_list_ns=[128,128,128,128,128,128,128,64,64,64,32,3]
        self.dec=nn.Sequential(decoder_general(channel_list_ns,name='ds'),nn.Hardtanh(-1,1))


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

        
        x=(((self.x+dx)/(self.x.shape[2]-1)-0.5)*2).squeeze(1)
        y=(((self.y+dy)/(self.y.shape[3]-1)-0.5)*2).squeeze(1)
        x,y=y,x

        grid=torch.cat([x.unsqueeze(3),y.unsqueeze(3)],3)

        return torch.nn.functional.grid_sample(inputs,grid,align_corners=False)



class decoder_bg(nn.Module):
    def __init__(self):
        global bottle
        super(decoder_bg,self).__init__()
        in_channels=16
        out_channels=1

        self.conv1=nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
                                nn.BatchNorm2d(out_channels),
                                nn.LeakyReLU(0.2))
        self.l1=nn.Sequential(nn.Linear(bottle,3),nn.Sigmoid())

        self.conv2=nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
                                nn.BatchNorm2d(out_channels),
                                nn.LeakyReLU(0.2))
        self.l2=nn.Sequential(nn.Linear(bottle,3),nn.Sigmoid())


    def forward(self,x,orig):
        global thresh
        ll=x.shape[1]//2
        x1=x[:,:ll,:,:]
        x=x[:,ll:,:,:]
        s=(torch.flatten(self.conv1(x),start_dim=1))
        s=self.l1(s)
        s_for=(self.l2(torch.flatten(self.conv2(x1),start_dim=1))+0.125)*0.8
        mask=torch.abs(s-s_for)
        mask1=mask.clone()
        mask[mask1>thresh]=0
        mask[mask1<=thresh]=1
        #mask[mask>thresh],mask[mask<=thresh]=0,1
        #print(mask[0],s[0],s_for[0])

        diff_for_loss=(1/torch.abs(s-s_for))*mask

        col=orig
        print("Foreground",s_for[0])
        print("Background",s[0])
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
        channel_list_ns=[128,128,128,128,128,128,128,64,64,64,32,3]
        self.dec=nn.Sequential(
            decoder_general(channel_list_ns,name='ns'),
            nn.Hardtanh(-0.01,0.01),
            )

    def forward(self,x):
        return self.dec(x)

class Conv_block_T(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_block_T, self).__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.net(x)


class decoder_general(nn.Module):
    def __init__(self,channel_list,name):
        
        super(decoder_general,self).__init__()
        self.u0=nn.Sequential(
            nn.ConvTranspose2d(in_channels=channel_list[0],out_channels=channel_list[1],kernel_size=2,stride=2),
            Conv_block(channel_list[1],channel_list[2]))
        #   2  ->   [128,8,8]
        #   1  ->   [256,4,4]
        self.u1 = nn.Sequential(
            Conv_block_T(channel_list[2]*2,channel_list[3]),
            Conv_block(channel_list[3],channel_list[4]),
            Conv_block(channel_list[4],channel_list[5]))
        #   5  ->   [128,16,16]
        self.u2 = nn.Sequential(
            Conv_block_T(channel_list[5]*2,channel_list[6]),
            Conv_block(channel_list[6],channel_list[7]),
            Conv_block(channel_list[7],channel_list[8]))
        #   8  ->   [64,32,32]
        self.u3 = nn.Sequential(
            Conv_block(channel_list[8]*2,channel_list[9]),
            #Conv_block(channel_list[9],channel_list[10]),
            Conv_block(channel_list[9],channel_list[11]))
        self.name=name
    def forward(self,store):
        #store['0'],store['1'],store['2']
        x=self.u0(store['encode_'+self.name])
        x=torch.cat([x,store['2']],dim=1)
        x=self.u1(x)
        x=torch.cat([x,store['1']],dim=1)
        x=self.u2(x)
        x=torch.cat([x,store['0']],dim=1)
        #print(x.shape)
        x=self.u3(x)
        return x

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        global padding_,kernel_

        self.d0=Conv_block(1, 64)
        self.d1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(64, 64),
            Conv_block(64, 128)
        )

        self.d2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(128, 128),
            Conv_block(128, 128),
        )

        self.d3=nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(128, 256)
        )

    def forward(self,x):
        x_0=self.d0(x)
        x_1 = self.d1(x_0)
        x_2 = self.d2(x_1)
        x_3 = self.d3(x_2)
        
        l1=x_3.shape[1]
        
        store={'encode_bg':x_3[:,:l1//8,:,:],'encode_ds':x_3[:,:(l1)//2,:,:],'encode_ns':x_3[:,((l1)//2):,:,:],'0':x_0,'1':x_1,'2':x_2}
        
        return store

class encode_mnist(nn.Module):
    def __init__(self):
        super(encode_mnist,self).__init__()
        self.encoder=encoder()
        channel_list_ns=[256,128,128,128,128,128,128,64,64,32,32,3]
        self.decoder_1=decoder_ds()
        self.decoder_2=decoder_bg()
        self.decoder_3=decoder_ns()

    def forward(self,x):
        #print(x.size())
        x_en=self.encoder(x)
        
        dec1=self.decoder_1(x_en,x)
        thresh=torch.min(dec1)+torch.max(dec1)/4.0
        dec1[dec1>thresh],dec1[dec1<=thresh]=1,0
        #print(torch.unique(dec1[0][0].detach()))

        dec2=self.decoder_2(x_en['encode_bg'],dec1)
        dec3=self.decoder_3(x_en)
        out_avg=(dec2['dec']+dec3)
        return out_avg, dec2['loss'],dec2
        # return out_avg,dec2['loss'],dec2



in_=torch.rand(64,1,32,32).cuda()
in_[in_>0.5]=1
in_[in_<=0.5]=0
encoded=encode_mnist().cuda()(in_)[0]
# print(encoded['encode'].shape)  #   [64,256,4,4]
# print(encoded['0'].shape)   #   [64,64,32,32]
# print(encoded['1'].shape)   #   [64,128,16,16]
# print(encoded['2'].shape)   #   [64,128,8,8]

print(encoded.shape)