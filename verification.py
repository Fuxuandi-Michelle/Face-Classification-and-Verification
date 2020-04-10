import csv
import configparser
import torch
import torch.nn as nn
from ResNet import CNN
from dataloader import *
import pickle

class CNNVerf(nn.Module):
    def __init__(self, cnn):
        super(CNNVerf, self).__init__()
        self.cnn = cnn
        self.cossim = nn.CosineSimilarity()

    def forward(self, input1, input2):
        _ = self.cnn.forward(input1)
        self.emb1 = self.cnn.emb_output
        _ = self.cnn.forward(input2)
        self.emb2 = self.cnn.emb_output
        self.score = self.cossim(self.emb1, self.emb2)
        return self.score

def loadVerfData(batch_size):
    modelFilename = '11-785hw2p2-s20/test_trials_verification_student.txt'
    root = '11-785hw2p2-s20/test_verification/'
    f = open(modelFilename, 'r')
    lines = f.readlines()
    img_list = []
    ids = []
    for string in lines:
        img = string.split(" ")
        ids.append([img[0], img[1][:-1]])
        img_list.append([os.path.join(root, img[0]),os.path.join(root, img[1][:-1])])

    trainset = verfImageDataset(img_list)
    dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)
    return ids, dataloader

def test_verf(cnnverf, dataloader, gpuFlag):
    cnnverf.eval()
    Y_pred=[]

    for x_batch1, x_batch2 in dataloader:
        if gpuFlag:
            x_batch1 = x_batch1.cuda(0)
            x_batch2 = x_batch2.cuda(0)

        y_pred = cnnverf.forward(x_batch1.float(),x_batch2.float())

        if gpuFlag:
            y_pred = y_pred.cpu()

        Y_pred.extend(y_pred.tolist())

        del x_batch1
        del x_batch2

    return Y_pred

def writeToVerf(ids, Y_pred):
    modelFilename = './submit_verf.csv'

    f = open(modelFilename, 'w')
    with f:

        writer = csv.writer(f)
        writer.writerow(('trial', 'score'))
        for i in range(len(ids)):
            writer.writerow((ids[i][0]+" "+ids[i][1], Y_pred[i]))

    print("Write verification File complete!")

if __name__ == '__main__':

    configPath = "./config.ini"
    config = configparser.ConfigParser()
    config.read(configPath)

    # _______ gpu configuration _____
    gpuAvaliable = torch.cuda.is_available()
    gpuFlag = gpuAvaliable and config.get('test', 'use_gpu') == "True"
    cudaID = None
    if gpuFlag:
        print("__ use gpu __")
        cudaID = int(config.get("train", "cuda_id"))

    # _______ load model _____
    channels = list(map(int, config.get('model', 'channels').split(',')))
    linears = list(map(int, config.get('model', 'linears').split(',')))
    feat_dim = int(config.get('model', 'feat_dim'))
    batch_size = int(config.get('test', 'batch_size'))
    blocks = list(map(int, config.get('model', 'blocks').split(',')))
    flatten = config.get('model', 'flatten') == 'True'

    with open('savedModel/label_dict.pickle', 'rb') as handle:
        b = pickle.load(handle)

    # target_dict = b['label2id']
    # label_dict = b['id2label']
    num_class = b['num_class']

    cnn = CNN(channels, blocks, linears, num_class, flatten, feat_dim, gpuFlag)
    path = config.get('path', 'savedModelpath') + config.get('test', 'model_path')
    cnn.load_state_dict(torch.load(path))
    cnnverf = CNNVerf(cnn)
    print("__ load saved model __")

    if gpuFlag:
        print("__ move models to GPU __")
        cnnverf = cnnverf.cuda(cudaID)

    ids, dataloader = loadVerfData(batch_size)
    Y_pred = test_verf(cnnverf, dataloader, gpuFlag)
    writeToVerf(ids, Y_pred)