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

from RNN import RNN

import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import KFold
from tqdm import tqdm
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


input_file = r"E:\yanxin\data\test_100.csv"  # 样本文件路径
model_file = r'E:\yanxin\model/model_0111_test100.pth'  # 模型保存路径
class_num = 5  # 分类类别数量
ts = 256  # 时间序列长度
'''超参数'''
EPOCH = 1500  # epoch
BATCH_SIZE = 32  # batch_size
LR = 0.001  # 初始学习率
output_file = r'E:\yanxin\data\accuracy.xlsx'  # 精度文件保存路径

if len(sys.argv) > 2:
    input_file = sys.argv[1]
    model_file = sys.argv[2]
    class_num = int(sys.argv[3])
    ts = int(sys.argv[4])
    EPOCH = int(sys.argv[5])
    BATCH_SIZE = int(sys.argv[6])
    LR = float(sys.argv[7])
    if len(sys.argv) > 8:
        output_file = sys.argv[8]


use_cuda = torch.cuda.is_available()
print('-----------------------------------------------------')
print('岩芯矿物智能识别模型训练')
print('-----------------------------------------------------')
print('Is CUDA available: ', use_cuda)
print('-----------------------------------------------------')
OA = []
real = []
predal = []

kf = KFold(n_splits=5)  # 交叉验证


print('读取训练数据...')
t0 = time.time()
data1 = pd.read_csv(input_file, header=None)
array = np.array([data1.values])
print('训练数据读取完毕，用时: %0.1f(s).' % (time.time() - t0))
print('-----------------------------------------------------')

dataloader = torch.from_numpy(array)
dataloader = dataloader.permute(1, 0, 2)  # 维度更换

for train, test in kf.split(dataloader):
    train_x = dataloader[train, :, :ts]
    train_y = dataloader[train, :, -1]
    test_x = dataloader[test, :, :ts]
    test_y = dataloader[test, :, -1]
    best_accuracy = 0

    dataset_train = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(
        dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)

    rnn = RNN(class_num=class_num, use_cuda=use_cuda).to(device)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR, weight_decay=0.00001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20, eta_min=0.000001)

    loss_func = nn.CrossEntropyLoss()
    print('开始训练...')
    for epoch in tqdm(range(EPOCH)):
        rnn.train()
        for step, (x, z_y) in enumerate(tqdm(train_loader)):

            x = x.to(device)
            x = x.permute(2, 0, 1)
            b_y = z_y[:, 0]
            b_y = b_y.to(device)
            b_x = x.view(ts, -1, 1)  # 数据维度为1
            b_x = torch.as_tensor(b_x, dtype=torch.float32).to(device)

            output = rnn(b_x)

            loss = loss_func(output, b_y.long())
            optimizer.zero_grad()

            loss.backward()
            nn.utils.clip_grad_norm_(
                rnn.parameters(), max_norm=15, norm_type=2)
            optimizer.step()
        scheduler.step()

        rnn.eval()

        x1 = test_x.to(device)
        x1 = x1.permute(2, 0, 1)
        x1 = x1.view(ts, -1, 1)  # 数据维度为1
        if use_cuda:
            x1 = x1.float().cuda()
        else:
            x1 = x1.float()
        y1 = test_y[:, 0]
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

        print("\nepoch %d, best accuracy now %g" % (epoch_b, best_accuracy))
        print("epoch %d info:" % epoch, "loss=%0.3f " % (loss.data.item()), "test_loss=%0.3f " % (
            test_loss.data.item()), "test_acc=%0.4f" % (accuracy1.item()))

    checkpoint = torch.load(model_file)
    rnn.load_state_dict(checkpoint['rnn'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1

    rnn.eval()
    x2 = test_x.to(device)
    x2 = x2.permute(2, 0, 1)
    x2 = x2.view(ts, -1, 1)   # 数据维度为1
    if use_cuda:
        x2 = x2.float().cuda()
    else:
        x2 = x2.float()
    y2 = test_y[:, 0]
    y2 = y2.to(device)

    y_out1 = rnn(x2)

    pred1 = torch.max(y_out1, 1)[1].data.squeeze()
    accuracy = sum(pred1 == y2) / float(y1.shape[0])

    print('OA:', accuracy)

    OA.append(accuracy)
    OA.append(loss.data)
    OA.append(ts)

    if use_cuda:
        a = test_y.cuda().tolist()
    else:
        a = test_y.tolist()
    a = np.array(a)
    real.append(a[:, 0])
    predal.append(pred1)

    datareal = pd.DataFrame(real)
    datapred = pd.DataFrame(predal)
    data_oa = pd.DataFrame(OA)

    if output_file is not None:
        writer = pd.ExcelWriter(output_file)
        datareal.to_excel(writer, 'real', float_format='%.5f')
        datapred.to_excel(writer, 'pred', float_format='%.5f')
        data_oa.to_excel(writer, 'oa', float_format='%.5f')

        writer.save()
