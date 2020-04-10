from loss import CenterLoss
import torch.nn.functional as F
from utils import *


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, stride=1):
        super(ResidualBlock, self).__init__()
        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels)
            )
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample != None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class CNN(nn.Module):
    def __init__(self, channels, blocks, linears, num_class, flatten = False, feat_dim=None, gpu_flag=True):
        super(CNN, self).__init__()

        self.channels = channels
        self.flatten = flatten
        if flatten:
            self.flatten_shape = calculateShape(self.channels)
            self.feat_dim = feat_dim
        else:
            self.flatten_shape = self.channels[-1]
            self.feat_dim = self.channels[-1]
        # self.linears = [self.flatten_shape] + linears + [num_class]
        self.linears = [self.feat_dim] + linears + [num_class]
        print(self.linears)
        self.num_class = num_class
        self.inchannels = 64
        self.blocks = blocks

        self._norm_layer = nn.BatchNorm2d

        self.cnn = self._make_cnn(self.channels, self.blocks)
        self.classifier = self._make_classifier(self.linears)

        # self.layers = []
        # self.layers.append(self._make_cnn(self.channels, self.blocks))
        # self.layers.append(self._make_classifier(self.linears))

        # embedding for center loss
        self.emb_layer = nn.Sequential(
            # nn.BatchNorm1d(self.linears[0]),
            nn.Linear(self.flatten_shape, self.feat_dim, bias=False),
        )
        self.emb_avg = nn.AdaptiveAvgPool2d((1, 1))

        self.closs_criterion = CenterLoss(self.num_class, self.feat_dim, use_gpu=gpu_flag)
        self.ce_criterion = nn.CrossEntropyLoss()

    def _make_cnn(self, channels, blocks):
        clayers = []
        normlayer = self._norm_layer
        for i in range(len(channels)):
            # self.clayers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=5, stride=1, bias=False, padding=2))
            if i <= 0:
                clayers.append(
                    nn.Conv2d(3, channels[i], kernel_size=5, stride=1, bias=False, padding=2))
                clayers.append(normlayer(channels[i]))
                clayers.append(nn.ReLU())
                # self.clayers.append(ResidualBlock(channels[i+1], channels[i+1]))
            elif i==1:
                clayers.append(self._make_layer(channels[i], blocks[i], normlayer))
            else:
                clayers.append(self._make_layer(channels[i],blocks[i], normlayer, stride=2))

        clayers = nn.Sequential(*clayers)
        return clayers

    def _make_layer(self, outchannels, blocks, normlayer, stride=1):
        reslayer = []
        reslayer.append(ResidualBlock(self.inchannels, outchannels, normlayer, stride=stride))
        self.inchannels = outchannels
        for _ in range(1, blocks):
            reslayer.append(ResidualBlock(self.inchannels, outchannels, normlayer))
        reslayer = nn.Sequential(*reslayer)
        return reslayer

    def _make_classifier(self, linears):
        classifier = []

        classifier.append(nn.BatchNorm1d(linears[0]))
        classifier.append(nn.ReLU())
        classifier.append(nn.Linear(linears[0], linears[-1]))

        classifier = nn.Sequential(*classifier)
        return classifier

    def forward(self, input):

        output = self.cnn(input)
        if self.flatten:
            output = output.flatten(start_dim=1)
            self.emb_output = self.emb_layer(output)
        else:
            output = self.emb_avg(output)
            self.emb_output = output.flatten(start_dim=1)

        self.label_output = self.classifier(self.emb_output)
        # self.label_output = self.classifier(self.emb_output)

        # output = F.avg_pool2d(output, [output.size(2), output.size(3)], stride=1)
        # self.emb_output = self.emb_closs(output)

        y_pred = torch.argmax(F.log_softmax(self.label_output, dim=1), dim=1)

        return y_pred

    def loss(self, target, lamda):

        self.labelloss = self.ce_criterion(self.label_output, target)
        self.closs = self.closs_criterion(self.emb_output, target)
        return self.labelloss + lamda * self.closs

