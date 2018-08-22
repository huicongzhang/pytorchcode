# -*- coding: UTF-8 -*-
import os
import pandas
dataroot = 'data/UTKFace/UTKFace'
filelist = os.listdir(dataroot)
imagename = []
age = []
gender =[]
for filename in filelist:
    file = str(filename)
    #print('\n{}'.format(file.split('_')))
    imagename.append(filename)
    age.append(file.split('_')[0])
    gender.append(file.split('_')[1])
dataframe = pandas.DataFrame({'imagename':imagename,'age':age,'gender':gender})
dataframe.to_csv("UTKFace.csv",sep=',')
