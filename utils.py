import torch.nn as nn
import time
import torch
# data initialization

def init_randn(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0,1)

def init_xavier(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight.data)

def calculateShape(channels, input=32, stride=2, padding=1, kernal=3):
    size = input
    padding_conv = 2
    padding_max = 1
    stride_max=2
    stride_conv=1
    kernal_conv=5
    kernal_max=3
    for i in range(len(channels)-1):
        #size = int((size + 2 * padding_conv - kernal_conv)/stride_conv)+1 # conv
        if i<=0:
            size = int((size + 2 * padding_conv - kernal_conv) / stride_conv) + 1  # conv
            #size = int((size + 2 * padding_max - kernal_max) / stride_max) + 1  # max
        else:
            size = int((size + 2 * padding_max - 3) / 2) + 1  # conv
            #size = int((size + 2 * padding_max - kernal_max) / stride_max) + 1  # max

    #size = int((size + 2 * padding_max - kernal_max) / stride_max) + 1  # max
    print(channels[-1]*(size**2))
    return channels[-1]*(size**2)

def saveModel(modelDic, epoch, type='cnn'):
    savedModelPath = './savedModel/'
    modelFilename = type + time.strftime('%Y%m%d_%H%M%S') + "_ep" + str(epoch) + ".pt"
    torch.save(modelDic, savedModelPath + modelFilename)
    print(modelFilename)
    return modelFilename