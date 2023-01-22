# -*- coding: utf-8 -*-
"""
样本预处理
1. 样本txt转换为csv文件
2. 打乱样本顺序
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Created on Thu Jul 29 11:25:34 2021
Reconstructed by wHy 2023.01.01
"""
import random
import sys
from tqdm import tqdm

sr_txt_sample_path = r"E:\yanxin\data\test_100.txt"  # 输入txt路径
out_csv_sample_path = r"E:\yanxin\data/test_100_230111.csv"  # 输出csv路径

if len(sys.argv) > 2:
    sr_txt_sample_path = sys.argv[1]
    out_csv_sample_path = sys.argv[2]

print("预处理开始...")
out = open(out_csv_sample_path, 'w')
lines = []
with open(sr_txt_sample_path, 'r') as infile:
    for line in infile:
        lines.append(line)  # 全部读入lines列表中
    random.shuffle(lines)  # 打乱顺序
    for line in tqdm(lines):
        out.write(line)  # 写入

print("预处理完毕")
