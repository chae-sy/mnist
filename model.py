import torch.nn as nn

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class MNISTNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # features: Conv→BN→ReLU 블록 하나만
        self.features = ConvBNReLU(1, 32, kernel_size=3, padding=1)
        # classifier: Conv2d 레이어 하나
        # 입력 feature map은 32채널, 크기 28×28 → kernel_size=28로 해 두면 1×1 출력
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=28, bias=True)

    def forward(self, x):
        x = self.features(x)           # [B,32,28,28]
        x = self.classifier(x)         # [B,10,1,1]
        return x.view(x.size(0), -1)   # [B,10]
