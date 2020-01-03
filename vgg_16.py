import torch
import torch.nn as nn
from thop import profile
from thop import clever_format
from torchsummary import summary

cfg = {'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],}
		
class VGG(nn.Module):
    def __init__(self, vgg_type, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = self.make_layers(cfg[vgg_type])
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.classifier = nn.Linear(512, num_classes)
        if init_weights:
            self.initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        
        return nn.Sequential(*layers)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = VGG('VGG16').to(device)
    # Two methods for calculating FLOPs and parameters
    input = torch.randn(1, 3, 32, 32).to(device)
    flops, params = profile(model, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print('FLOPs:', flops)
    print('Parameters:', params)
    summary(model, (3, 32, 32), batch_size=1) 

