'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torchsummary


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    # function to extact the multiple features
    def feature_list(self, x):
        out_list = []
        sequential_len = len(self.features)
        out = self.features[0](x)
        for i in range(1, sequential_len):
            out = self.features[i](out)
            if self.features[i]._get_name() == "ReLU":
                out_list.append(out)
        out = out.view(out.size(0), -1)
        # out_list.append(out)
        y = self.classifier(out)
        for i in range(len(out_list)):
            out_list[i] = out_list[i].view(out_list[i].size(0), out_list[i].size(1), -1)
            out_list[i] = torch.mean(out_list[i].data, 2)
        return y, out_list


if __name__ == '__main__':
    net = VGG('VGG16')
    x = torch.randn(1,3,32,32)
    y, out_features = net.feature_list(x)
    num = len(out_features)
    for i in range(num):
        print(out_features[i].shape)
    torchsummary.summary(net.cuda(), (3, 32, 32))
