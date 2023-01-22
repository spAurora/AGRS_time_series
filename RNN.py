"""
LSTM
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

D_num = 1  # 输入数据的维度
layer_num = 2  # 网络层数 1-5
N_num = 64  # 隐藏层中的神经元数量2-80，2-60
dropout = 0.5  # 随机失活


class RNN(nn.Module):

    def __init__(self, class_num=5, use_cuda=True):
        super(RNN, self).__init__()

        self.use_cuda = use_cuda

        self.lstm = nn.LSTM(
            input_size=D_num,
            hidden_size=N_num,
            num_layers=layer_num,
            batch_first=False,
            dropout=dropout
        )

        self.fc1 = nn.Sequential(
            nn.Linear(N_num, N_num),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(N_num),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(N_num, class_num),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm1d(class_num),
        )

    def forward(self, x):
        if self.use_cuda:  # GPU模式
            h0 = Variable(torch.zeros(layer_num, x.size(1), N_num)).cuda()
            c0 = Variable(torch.zeros(layer_num, x.size(1), N_num)).cuda()
        else:  # CPU模式
            h0 = Variable(torch.zeros(layer_num, x.size(1), N_num))
            c0 = Variable(torch.zeros(layer_num, x.size(1), N_num))

        lstm_out, (h, c) = self.lstm(x, (h0, c0))

        y1 = self.fc1(lstm_out[-1, :, :])
        y2 = self.fc2(y1)

        return y2
