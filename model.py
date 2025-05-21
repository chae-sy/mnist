import torch
import torch.nn as nn

class SubSpectralNorm(nn.Module):
    def __init__(self, num_features, spec_groups=4, affine="Sub", batch=True, dim=2):
        super().__init__()
        self.spec_groups = spec_groups
        self.affine_all = False
        affine_norm = False
        if affine == "Sub":
            affine_norm = True
        elif affine == "All":
            self.affine_all = True
            self.weight = nn.Parameter(torch.ones((1, num_features, 1, 1)))
            self.bias = nn.Parameter(torch.zeros((1, num_features, 1, 1)))
        if batch:
            self.ssnorm = nn.BatchNorm2d(num_features * spec_groups, affine=affine_norm)
        else:
            self.ssnorm = nn.InstanceNorm2d(num_features * spec_groups, affine=affine_norm)
        self.sub_dim = dim

    def forward(self, x):
        if self.sub_dim in (3, -1):
            x = x.transpose(2, 3).contiguous()
        b, c, h, w = x.size()
        assert h % self.spec_groups == 0, f"Height {h} not divisible by spec_groups {self.spec_groups}"
        x = x.view(b, c * self.spec_groups, h // self.spec_groups, w)
        x = self.ssnorm(x)
        x = x.view(b, c, h, w)
        if self.affine_all:
            x = x * self.weight + self.bias
        if self.sub_dim in (3, -1):
            x = x.transpose(2, 3).contiguous()
        return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), groups=out_channels, bias=False),
            SubSpectralNorm(out_channels, spec_groups=4),  # 28 divisible by 4
            nn.ReLU(inplace=True)
        )


class MNISTNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = ConvBNReLU(1, 32, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=(28, 28), bias=True)

    def forward(self, x):
        x = self.features(x)  # [B, 32, 28, 28]
        x = self.classifier(x)  # [B, 10, 1, 1]
        return x.view(x.size(0), -1)  # [B, 10]

