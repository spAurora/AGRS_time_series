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
import sys
import pandas as pd
import numpy as np
import math
from RNN import RNN
from tqdm import tqdm
import time
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sr_data_path = r"E:\yanxin\data/sr_img.csv"  # 待预测数据路径
model_path = r'E:\yanxin/model\model_0717_test100.pth'  # 模型路径
output_path = r'E:\yanxin\predict\predict_test_100.xlsx'  # 预测结果输出路径
class_num = 5  # 分类类别数量
ts = 256  # 时间序列长度

if len(sys.argv) > 2:
    sr_data_path = sys.argv[1]
    model_path = sys.argv[2]
    output_path = sys.argv[3]
    class_num = int(sys.argv[4])
    ts = int(sys.argv[5])

print('-----------------------------------------------------')
print('岩芯矿物智能识别模型预测')
print('-----------------------------------------------------')

use_cuda = torch.cuda.is_available()
print('Is CUDA available: ', use_cuda)
print('-----------------------------------------------------')

print('读取待预测数据...')
t0 = time.time()
data1 = pd.read_csv(sr_data_path, header=None)
print('读取待预测数据完毕，用时: %0.1f(s).' % (time.time() - t0))
print('-----------------------------------------------------')

array = np.array([data1.values])
dataloader = torch.from_numpy(array)
dataloader = dataloader.permute(1, 0, 2)

rnn = RNN(class_num=class_num, use_cuda=use_cuda).to(device)
if use_cuda:
    checkpoint = torch.load(model_path)
else:
    checkpoint = torch.load(model_path, map_location='CPU')
rnn.load_state_dict(checkpoint['rnn'])
rnn.eval()

test = dataloader[:, :, :ts]
i = math.ceil(test.shape[0] / 1000)

print('开始预测...')
t0 = time.time()
predal = []
for n in tqdm(range(i)):
    fn = n * 1000
    ln = fn + 1000
    testto2 = test[fn:ln, :, :]

    x2 = testto2.to(device)
    x2 = x2.permute(2, 0, 1)
    x2 = x2.view(ts, -1, 1)
    if use_cuda:
        x2 = x2.float().cuda()
    else:
        x2 = x2.float()
    with torch.no_grad():
        y_out1 = rnn(x2)
        pred1 = torch.max(y_out1, 1)[1].data.cpu().numpy().squeeze()
        predal.append(pred1)
print('预测完毕，用时: %0.1f(s).' % (time.time() - t0))
print('-----------------------------------------------------')

print('写入预测结果...')
t0 = time.time()
datapred = pd.DataFrame(predal)
writer = pd.ExcelWriter(output_path)  # 预测的类别
datapred.to_excel(writer, 'pred', float_format='%.5f')
writer.save()
print('写入完毕，用时: %0.1f(s).' % (time.time() - t0))
