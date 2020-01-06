import torch
import torch.nn as nn
from thop import profile
from thop import clever_format
from torchsummary import summary

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.leaky(out)

        return out

class ResBlock(nn.Module):
    # Residual block. The residual path has an 1x1 and 3x3 convolution blocks.
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.layer1 = ConvBlock(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0)
        self.layer2 = ConvBlock(in_channels//2, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        return x + out

class Darknet53(nn.Module):
    def __init__(self, num_classes=10, init_weights=True):
        super(Darknet53, self).__init__()
        self.conv1 = ConvBlock(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)
        self.res1 = self.make_layer(in_channels=64, num_blocks=1)
        self.conv3 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.res2 = self.make_layer(in_channels=128, num_blocks=2)
        self.conv4 = ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)
        self.res3 = self.make_layer(in_channels=256, num_blocks=8)
        self.conv5 = ConvBlock(256, 512, kernel_size=3, stride=2, padding=1)
        self.res4 = self.make_layer(in_channels=512, num_blocks=8)
        self.conv6 = ConvBlock(512, 1024, kernel_size=3, stride=2, padding=1)
        self.res5 = self.make_layer(in_channels=1024, num_blocks=4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, num_classes)

        if init_weights:
            self.initialize_weights()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out)
        out = self.conv3(out)
        out = self.res2(out)
        out = self.conv4(out)
        out = self.res3(out)
        out = self.conv5(out)
        out = self.res4(out)
        out = self.conv6(out)
        out = self.res5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(ResBlock(in_channels))

        return nn.Sequential(*layers)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Darknet53().to(device)
    # Two methods for calculating FLOPs and parameters
    input = torch.randn(1, 3, 608, 608).to(device)
    flops, params = profile(model, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print('FLOPs:', flops)
    print('Parameters:', params)
    
    summary(model, (3, 608, 608), batch_size=1)

