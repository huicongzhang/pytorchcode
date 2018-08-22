# -*- coding: UTF-8 -*-
import os

import numpy as np
import pandas
import torch
import torch.optim
from skimage import io,transform
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class CusDataset(Dataset):
    def __init__(self,csv_file,root_dir,Wem,transform=None):
        """
        arg:
            csv_file(string):数据集标签的文件路径
            root_dir(string):图片路径
            Wem(int):0是男，1是女
            transform(optional):图像变换方法
        """
        self.labels = pandas.read_csv(csv_file)
        image = []
        age = []
        gender = []
        temp = 0
        for i in range(len(self.labels)):
            if self.labels.iloc[i, 3] == Wem:
                image.append(self.labels.iloc[i, 1])
                if self.labels.iloc[i, 2] >= 0 and self.labels.iloc[i, 2] <5:
                    temp = 0
                elif self.labels.iloc[i, 2] >= 5 and self.labels.iloc[i, 2] <10:
                    temp = 1
                elif self.labels.iloc[i, 2] >= 10 and self.labels.iloc[i, 2] <15:
                    temp = 2
                elif self.labels.iloc[i, 2] >= 15 and self.labels.iloc[i, 2] <20:
                    temp = 3
                elif self.labels.iloc[i, 2] >= 20 and self.labels.iloc[i, 2] <30:
                    temp = 4
                elif self.labels.iloc[i, 2] >= 30 and self.labels.iloc[i, 2] <40:
                    temp = 5
                elif self.labels.iloc[i, 2] >= 40 and self.labels.iloc[i, 2] <50:
                    temp = 6
                elif self.labels.iloc[i, 2] >= 50 and self.labels.iloc[i, 2] <60:
                    temp = 7
                elif self.labels.iloc[i, 2] >= 60 and self.labels.iloc[i, 2] <70:
                    temp = 8
                elif self.labels.iloc[i, 2] >= 70:
                    temp = 9
                #aget = np.zeros([10])
                #aget = temp
                age.append(temp)
                gender.append(self.labels.iloc[i, 3])
            else:
                continue
        self.labels = {'imagename':image,'age':age,'gender':gender}
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.labels['imagename'])
    def __getitem__(self,idx):
        
        img_name = os.path.join(self.root_dir,
                                self.labels['imagename'][idx])
        image = io.imread(img_name)
        image = transform.resize(image,output_shape=(256,256))
        age = self.labels['age'][idx]
        gender = self.labels['gender'][idx]
        sample = {'imagename':image,'age':age,'gender':gender}
        if self.transform:
            image = self.transform(sample['imagename'])
        sample = {
                    'imagename':image,
                    'age':torch.from_numpy(np.array([age])),
                    'gender':torch.from_numpy(np.array([gender]))
                 }
        return sample
class ToTensor(object):
    """将ndarrays的样本转化为Tensors的样本"""
    def __call__(self, sample):
        image,age,gender = sample['imagename'], sample['age'],sample['gender']
        # 交换颜色通道, 因为
        # numpy图片: H x W x C
        # torch图片   : C X H X W
        image = image.transpose((2, 0, 1))
        return {'imagename': torch.from_numpy(image),
                'age': torch.from_numpy(np.array([age])),
                'gender':torch.from_numpy(np.array([gender]))
                }
if __name__ == '__main__':
    face_dataset = CusDataset(
        csv_file='../UTKFace.csv',
        root_dir='../data/UTKface/UTKface',
        Wem = 0,
        transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                ]
            )
    )
    dataloader = DataLoader(
        face_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4
    )
    m_face = iter(face_dataset)
    sample_batched = m_face.__next__()
    print(sample_batched['age'])
    #for i_batch,sample_batched in enumerate(dataloader):
    print("\nimagesize:{},age:{},gender:{}".format(sample_batched['imagename'].size(),sample_batched['age'],sample_batched['gender']))
