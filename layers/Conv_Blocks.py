import torch
import torch.nn as nn


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layer_scale_init_value=1e-6, init_weight=True):
        super(ConvNeXtBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Depthwise Convolutions (grouped convolutions)
        self.depthwise_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
            for _ in range(2)  # Assuming two layers of depthwise convolutions
        ])

        # 1x1 Convolutions for dimensionality reduction and restoration
        self.projection_conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.projection_conv2 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1)

        # GELU activation
        self.gelu = nn.GELU()

        # Layer Scale parameter
        self.gamma = nn.Parameter(torch.ones(out_channels) * layer_scale_init_value)

        # Layer Normalization
        self.norm = nn.LayerNorm(out_channels)

        # Dropout
        self.dropout = nn.Dropout(0.2)  # Assuming a dropout rate of 0.2

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Depthwise convolutions
        for conv in self.depthwise_convs:
            x = conv(x)
            x = self.norm(x)
            x = self.gelu(x)

        # Dimensionality reduction
        x = self.projection_conv1(x)
        x = self.gelu(x)

        # Second set of Depthwise convolutions (optional, based on actual ConvNeXt design)
        for conv in self.depthwise_convs:
            x = conv(x)
            x = self.norm(x)
            x = self.gelu(x)

        # Dimensionality restoration
        x = self.projection_conv2(x)

        # Layer Scale
        x = x * self.gamma

        # Dropout
        x = self.dropout(x)

        # Add residual connection
        x = x + residual
        return x