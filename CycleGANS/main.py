# -*- coding: UTF-8 -*-
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.utils as vutils
from dataset import CusDataset
from model import D, G
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def to_cuda(data):
    if torch.cuda.is_available():
        data = data.cuda()
    return data

def train(config):
    face_M_dataset = CusDataset(
        csv_file=config.csv_dir,
        root_dir=config.root_dir,
        Wem = 0,
        transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                ]
            )
    )
    face_W_dataset = CusDataset(
        csv_file=config.csv_dir,
        root_dir=config.root_dir,
        Wem = 1,
        transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                ]
            )
    )
    M_loader = DataLoader(
        face_M_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    W_loader = DataLoader(
        face_W_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    Dw = D()
    Dm = D()
    Gwm = G()
    Gmw = G()
    if torch.cuda.is_available():
            print('cuda')
            Dw = Dw.cuda()
            Dm = Dm.cuda()
            Gwm = Gwm.cuda()
            Gmw = Gmw.cuda()
    g_params = list(Gwm.parameters())+list(Gmw.parameters())
    d_params = list(Dw.parameters())+list(Dm.parameters())
    #print(d_params)
    g_optimer = optim.Adam(g_params,lr=config.lr)
    d_optimer = optim.Adam(d_params,lr=config.lr)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    writer = SummaryWriter()
    global_step = 0
    for i in range(config.epoch):
        M_data = iter(M_loader)
        W_data = iter(W_loader)
        while True:
            try:
                M_batched = M_data.__next__()
                W_batched = W_data.__next__()
            except StopIteration:
                break
            M_image = M_batched['imagename'].type(torch.cuda.FloatTensor)
            M_label = M_batched['age'].type(torch.cuda.LongTensor).squeeze()
            W_image = W_batched['imagename'].type(torch.cuda.FloatTensor)
            W_label = W_batched['age'].type(torch.cuda.LongTensor).squeeze()
            
            if W_label.size(0) != M_label.size(0):
                break
            fake_label = np.zeros([W_label.size(0)])
            fake_label[:] = 10
            #fake_label[:,10] = 1
            fake_label = torch.from_numpy(fake_label)
            fake_label = fake_label.type(torch.cuda.LongTensor)
            #print(M_label.type())
            '''print(fake_label.size())
            print("\n{}".format(W_label.view(W_label.size(0)*10)))
            '''
            #-----------------------------train D---------------------------
            #-------------train real----------
            g_optimer.zero_grad()
            d_optimer.zero_grad()

            dm_real_loss = criterion(Dm(M_image),M_label)
            dw_real_loss = criterion(Dw(W_image),W_label)
            d_real_loss = dm_real_loss+dw_real_loss
            d_real_loss.backward()
            d_optimer.step()
            #-------------train fake-----------
            g_optimer.zero_grad()
            d_optimer.zero_grad()

            dm_fake_loss = criterion(Dm(Gwm(W_image)),fake_label)
            dw_fake_loss = criterion(Dw(Gmw(M_image)),fake_label)
            d_fake_loss = dm_fake_loss+dw_fake_loss
            d_fake_loss.backward()
            d_optimer.step()
            #-----------------------------train G-----------------------------
            #----------------train--W-M-W-Cycle-------------
            g_optimer.zero_grad()
            d_optimer.zero_grad()

            fake_m_image = Gwm(W_image)
            g_WMW_loss = criterion(Dm(fake_m_image),M_label) + torch.mean(((W_image-Gmw(fake_m_image))**2))
            g_WMW_loss.backward()
            g_optimer.step()
            #----------------train--M-W-M-Cycle-------------
            g_optimer.zero_grad()
            d_optimer.zero_grad()

            fake_w_image = Gmw(M_image)
            g_MWM_loss = criterion(Dw(fake_w_image),W_label) + torch.mean(((M_image-Gwm(fake_w_image))**2))
            g_MWM_loss.backward()
            g_optimer.step()
            fake_w_image =  vutils.make_grid(fake_w_image, normalize=False, scale_each=False)
            fake_m_image =  vutils.make_grid(fake_m_image, normalize=False, scale_each=False)
            writer.add_scalar(tag='loss/g_MWM_loss',scalar_value=g_MWM_loss,global_step=global_step)
            writer.add_scalar(tag='loss/g_WMW_loss',scalar_value=g_WMW_loss,global_step=global_step)
            writer.add_scalar(tag='loss/d_fake_loss',scalar_value=d_fake_loss,global_step=global_step)
            writer.add_scalar(tag='loss/d_real_loss',scalar_value=d_real_loss,global_step=global_step)
            writer.add_image('iamge/fake_w_image',fake_w_image,global_step)
            writer.add_image('iamge/fake_m_image',fake_m_image,global_step)
            print("{}\n".format(global_step))
            global_step += 1
    writer.close()
            
    '''for i_batch,sample_batched in enumerate(dataloader):
        print("imagesize:{}\n".format(sample_batched['imagename'].size()))
        out = G12(sample_batched['imagename'].float())
        print(out.size())'''
def main(config):
    if config.mode == 'train':
        print('trian\n')
        train(config)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #训练参数设置
    parser.add_argument('--epoch',type=int,default=10)
    parser.add_argument('--batch_size',type=int,default=10)
    parser.add_argument('--lr',type=float,default=0.0002)

    #保存参数设置
    parser.add_argument('--mode',type=str,default='train')
    parser.add_argument('--root_dir',type=str,default='/home/zhc/Image/UTKFace/UTKFace')
    parser.add_argument('--csv_dir',type=str,default='../UTKFace.csv')
    parser.add_argument('--log_dir',type=str,default='./log')
    
    config = parser.parse_args()
    print(config)
    main(config)
