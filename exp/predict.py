from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np
import argparse
import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

def data_provider(path, flag):
    Data = UEAloader
    if flag == 'test':
        shuffle_flag = False
        batch_size = 16
        drop_last = False
        data_set = Data(
            root_path=path,
            flag=flag,
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=0,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=197)
        )
    return data_set, data_loader

paths =['','','']
for path in paths:
    flag='test'
    test_data, test_loader = data_provider(path,flag)
    print('test_data:', len(test_data), type(test_data))
    print('*****', test_data[0][0][0])

    def extract_doy_ndvi(rowcol_DOY_NDVI):
        elements = rowcol_DOY_NDVI.split(',')
        # 检查元素数量，如果少于3个，则跳过处理
        if len(elements) < 3:
            return None
        if not elements[2].strip():
            return None
        row = rowcol_DOY_NDVI.split(',')[0]
        row = float(row) * 1000000
        col = rowcol_DOY_NDVI.split(',')[1]
        col = float(col) * 1000000

        rowcol = str(int(row)) + '_' + str(int(col))
        return rowcol

    def readDoyNDVI(filepath):
        with open(filepath, "r") as f:  # 打开文件
            lrowCol_doy_ndvi = f.readlines()[9:]  # 读取文件
        l_rc_doy_ndvi = list(filter(None, map(extract_doy_ndvi, lrowCol_doy_ndvi)))
        return l_rc_doy_ndvi

    rowcol = readDoyNDVI(path)
    model = model_dict['TimesNet'].Model(self.args).float()

    model.load_state_dict(torch.load(os.path.join(
            './checkpoints/classification_A0421_94_TimesNet_UEA_ftM_sl197_ll1_pl0_dm128_nh9_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_test_0/',
            'checkpoint.pth')))

    preds = []
    trues = []
    folder_path = './test_results/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    self.model.eval()
    with torch.no_grad():
        for i, (batch_x, label, padding_mask) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            padding_mask = padding_mask.float().to(self.device)
            label = label.to(self.device)

            outputs = self.model(batch_x, padding_mask, None, None)

            preds.append(outputs.detach())
            trues.append(label)

    preds = torch.cat(preds, 0)
    trues = torch.cat(trues, 0)
    print('test shape:', preds.shape, trues.shape)

    # 保存每个样本的预测标签和概率到文
    # 计算每个类别的概率
    probs = torch.nn.functional.softmax(preds, dim=1)  # (total_samples, num_cl

    # 获取每个样本最可能的类别的索引
    max_prob_indices = torch.argmax(probs, dim=1)  # (total_samples,)

    # 使用 torch.gather 来获取每个样本最可能的类别的最大概率
    # probs 的第一个维度是样本，第二个维度是类别
    max_probs = torch.gather(probs, 1, max_prob_indices.unsqueeze(1)).squeeze(1)  # (total_samples,)

    # 接下来，您可以继续您的代码，例如计算准确率等
    # ...
    # 获取最可能的类别索引（即预测标签）
    predictions = torch.argmax(preds, dim=1).cpu().numpy()  # (total_samples,)

    # 将 trues 展平成一维张量，并转换为 numpy 数组
    # trues = trues.flatten().cpu().numpy()

    # 计算准确率
    # accuracy = cal_accuracy(predictions, trues)

    # 将预测概率转换为 numpy 数组，用于输出
    # 注意：这里需要指定具体的概率索引，或者使用其他方式获取所有样本的概率
    probs = max_probs.cpu().numpy()
    print(f'Length of rowcol: {len(rowcol)}')
    print(f'Length of max_probs: {len(max_probs)}')

    # 打印预测标签数组的长度
    print(f'Length of predictions: {len(predictions)}')

    # 保存每个样本的预测标签和概率到文件
    # 假设 rowcol 是一个与 predictions 和 probs 长度相同的列表
    predictions_and_probs = list(zip(rowcol, predictions, probs))

    # result save
    folder_path = './results/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(os.path.join(folder_path, 'predictions_and_probs15.txt'), 'w') as f:
        for rowcols, label, prob in predictions_and_probs:
            # 将概率列表中的每个元素转换为字符串，然后用逗号和空格分隔
            f.write('{},{},{} \n'.format(rowcols, label, str(prob)))
    print('============over===============')
    # with open(os.path.join(folder_path, 'accuracy.txt'), 'w') as f:
    #     f.write('accuracy:{}'.format(accuracy))
    # print('accuracy:{}'.format(accuracy))

    return