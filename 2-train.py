# -*- coding: utf-8 -*-

"""
LSTM
模型训练
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
751984964@qq.com
"""
import torch
import torch.nn as nn
import torch.utils.data as Data

import pandas as pd 
import numpy as np

from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from torch.autograd import Variable

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

input_file = r"E:\yanxin\data\test_100.csv" # 样本文件路径
output_file =r'E:\yanxin\data\accuracy.xlsx' # 精度文件保存路径
model_file = r'E:\yanxin\model/model_0103_test100.pth' # 模型保存路路径
class_num = 5 # 分类类别数量
ts = 256  # 时间序列长度
D_num = 1 # 输入数据的维度
layer_num =2 # 网络层数 1-5
N_num = 64 # 隐藏层中的神经元数量2-80，2-60
dropout = 0.5 # 随机失活

'''超参数'''
EPOCH = 1500 # epoch
BATCH_SIZE = 32 # batch_size
LR = 0.001 # 初始学习率


use_cuda = torch.cuda.is_available()
print(use_cuda)

OA = []
real = []
predal = []

kf = KFold(n_splits = 5)

data1 = pd.read_csv(input_file, header=None)  
array = np.array([data1.values])

dataloader = torch.from_numpy(array)
dataloader = dataloader.permute(1,0,2)

for train,test in kf.split(dataloader):
    train_x = dataloader[train,:,:ts]
    train_y = dataloader[train,:,-1]
    test_x = dataloader[test,:,:ts]
    test_y = dataloader[test,:,-1]
    best_accuracy = 0

    dataset_train = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)

    rnn = RNN().to(device)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR, weight_decay=0.00001) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.000001)

    loss_func = nn.CrossEntropyLoss() 

    for epoch in range(EPOCH):
        rnn.train()

        for step, (x, z_y) in enumerate(train_loader):  

            x=x.to(device)
            x = x.permute(2,0,1)
            b_y = z_y[:,0]
            b_y=b_y.to(device)
            b_x = x.view(ts, -1, D_num) 
            b_x = torch.as_tensor(b_x, dtype=torch.float32).to(device)

            output = rnn(b_x) 

            loss = loss_func(output, b_y.long())  
            optimizer.zero_grad()  

            loss.backward() 
            nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=15, norm_type=2)
            optimizer.step()
        scheduler.step()

        rnn.eval()

        x1 = test_x.to(device)
        x1 = x1.permute(2,0,1)
        x1 = x1.view(ts,-1,D_num)
        if use_cuda:
            x1 = x1.float().cuda()
        else:
            x1 = x1.float()
        y1 = test_y[:,0]
        y1 = y1.to(device)
        
        y_out = rnn(x1)

        pred = torch.max(y_out, 1)[1].data.squeeze()
        accuracy1 = sum(pred == y1) / float(y1.shape[0])
        test_loss = loss_func(y_out, y1.long())

        
        if accuracy1 >= best_accuracy:
            epoch_b = epoch
            state = {
                'epoch': epoch,
                'rnn': rnn.state_dict(),
                'optimizer': optimizer.state_dict(),
                }
            torch.save(state, model_file)
            best_accuracy = accuracy1
            
        print("epoch %d, best accuracy %g" % (epoch_b, best_accuracy))
        print("epoch = ",epoch,"step=",step," loss = ",loss.data,"test_loss = ",test_loss.data,"test_acc = ",accuracy1)

    checkpoint = torch.load(model_file)
    rnn.load_state_dict(checkpoint['rnn'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch =  checkpoint['epoch'] + 1

    rnn.eval()
    x2 = test_x.to(device)
    x2 = x2.permute(2,0,1)
    x2 = x2.view(ts,-1,D_num)
    if use_cuda:
        x2 = x2.float().cuda()
    else:
        x2 = x2.float()
    y2 = test_y[:,0]
    y2 = y2.to(device)

    y_out1 = rnn(x2)

    pred1 = torch.max(y_out1, 1)[1].data.squeeze()
    accuracy = sum(pred1 == y2) / float(y1.shape[0])

    print('OA:',accuracy)

    OA.append(accuracy)
    OA.append(loss.data)
    OA.append(ts)

    if use_cuda:
        a = test_y.cuda().tolist()
    else:
        a = test_y.tolist()
    a=np.array(a)
    real.append(a[:,0])
    predal.append(pred1)

    datareal = pd.DataFrame(real)
    datapred = pd.DataFrame(predal)
    data_oa = pd.DataFrame(OA)

    writer = pd.ExcelWriter( output_file )
    datareal.to_excel(writer,'real',float_format='%.5f')
    datapred.to_excel(writer,'pred',float_format='%.5f')
    data_oa.to_excel(writer,'oa',float_format='%.5f')

    writer.save()