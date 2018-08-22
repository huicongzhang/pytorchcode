# -*- coding: UTF-8 -*-
import torch
import torch.optim
from torchvision import datasets,transforms
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)
class G(nn.Module):
    def __init__(self, conv_dim=64):
        super(G, self).__init__()
        # encoding blocks
        self.conv1 = conv(3,conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2,conv_dim*4,4)
        # residual blocks
        self.conv4 = conv(conv_dim*4, conv_dim*4, 3, 1, 1)
        self.conv5 = conv(conv_dim*4, conv_dim*4, 3, 1, 1)
        self.conv5 = conv(conv_dim*4, conv_dim*4, 3, 1, 1)
        # decoding blocks
        self.deconv1 = deconv(conv_dim*4, conv_dim*2, 4)
        self.deconv2 = deconv(conv_dim*2,conv_dim,4)
        self.deconv3 = deconv(conv_dim, 3, 4, bn=False)
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)      # (-1, 64,128,128)
        out = F.leaky_relu(self.conv2(out), 0.05)    # (-1, 128,64,64)
        out = F.leaky_relu(self.conv3(out), 0.05)    #(-1, 256,32,32)

        out = F.leaky_relu(self.conv4(out), 0.05)    # ( " )
        out = F.leaky_relu(self.conv5(out), 0.05)    # ( " )
        
        out = F.leaky_relu(self.deconv1(out), 0.05)  # (-1, 128, 64, 64)
        out = F.leaky_relu(self.deconv2(out), 0.05)  # (-1, 64, 128, 128)
        out = F.tanh(self.deconv3(out))              # (-1, 3, 256, 256)
        return out
class D(nn.Module):
    """Discriminator"""
    def __init__(self, conv_dim=64):
        super(D, self).__init__()
        self.conv1 = conv(3,conv_dim, 4, bn=False)  # (-1,64,128,128)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)  # (-1,128,64,64)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)# (-1,256,32,32)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4)# (-1,512,16,16)
        self.conv5 = conv(conv_dim*8, conv_dim*8, 4)# (-1,512,8,8)
        self.conv6 = conv(conv_dim*8, conv_dim*8, 4)#(-1,512,4,4)
        self.fc = conv(conv_dim*8,11,4,1,0)
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)    
        out = F.leaky_relu(self.conv2(out), 0.05)  
        out = F.leaky_relu(self.conv3(out), 0.05)  
        out = F.leaky_relu(self.conv4(out), 0.05)    
        out = F.leaky_relu(self.conv5(out), 0.05)  
        out = F.leaky_relu(self.conv6(out), 0.05)
        out = self.fc(out).squeeze()
        return out






