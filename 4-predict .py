# -*- coding: utf-8 -*-

"""
LSTM
模型预测
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
751984964@qq.com
"""
import torch
import torch.nn as nn
import torch.utils.data as Data
import os,sys
import pandas as pd 
import numpy as np
from skimage import io
import math
from osgeo import gdal
from osgeo import ogr
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from torch.autograd import Variable
from torch.optim import lr_scheduler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    
    def __init__(self):
        super(RNN, self).__init__()

        self.lstm = nn.LSTM(
                input_size=D_num,
                hidden_size=N_num,
                num_layers=layer_num,
                batch_first = False,
                dropout=dropout
            )

        self.fc1 = nn.Sequential(
            nn.Linear(N_num, N_num),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(N_num),                
        )

        self.fc2 = nn.Sequential(
            nn.Linear(N_num, class_num),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm1d(class_num),                
        )
        
    def forward(self, x):
        if use_cuda: 
            h0 = Variable(torch.zeros( layer_num,x.size(1), N_num)).cuda()
            c0 = Variable(torch.zeros( layer_num,x.size(1), N_num)).cuda()
        else:
            h0 = Variable(torch.zeros( layer_num,x.size(1), N_num))
            c0 = Variable(torch.zeros( layer_num,x.size(1), N_num))

        lstm_out, (h, c) = self.lstm(x, (h0,c0))

        y1 = self.fc1(lstm_out[-1,:,:])
        y2 = self.fc2(y1)

        return y2

sr_data_path = r"E:\yanxin\data/sr_img.csv" # 待预测数据路径
model_path = r'E:\yanxin/model\model_0717_test100.pth' # 模型路径
output_path = r'E:\yanxin\predict\predict_test_100.xlsx' # 预测结果输出路径
class_num = 5 # 分类类别数量
ts = 256 # 时间序列长度
D_num = 1 # 输入数据的维度
layer_num = 2 # 网络层数 1-5
N_num = 64 # 隐藏层中的神经元数量2-80，2-60
dropout = 0.5 # 随机失活

use_cuda = torch.cuda.is_available()
print(use_cuda)

data1 = pd.read_csv(sr_data_path, header=None)
array = np.array([data1.values])
dataloader = torch.from_numpy(array)
dataloader = dataloader.permute(1,0,2)

rnn = RNN().to(device)
checkpoint = torch.load(model_path)
rnn.load_state_dict(checkpoint['rnn'])
rnn.eval()

test = dataloader[:,:,:ts]
i = math.ceil(test.shape[0]/10000)

predal = []
for n in range(i):
    fn=n*10000
    ln=fn+10000
    testto2 = test[fn:ln,:,:]

    x2 = testto2.to(device)
    x2 = x2.permute(2,0,1)
    x2 = x2.view(ts,-1,D_num)
    if use_cuda:
        x2 = x2.float().cuda()
    else:
        x2 = x2.float()
    y_out1 = rnn(x2)
    pred1 = torch.max(y_out1, 1)[1].data.cpu().numpy().squeeze()
    predal.append(pred1)

datapred = pd.DataFrame(predal)

writer = pd.ExcelWriter(output_path) # 预测的类别
datapred.to_excel(writer,'pred',float_format='%.5f')
writer.save()







