import configparser
from dataloader import *
from ResNet import CNN
import numpy as np
from utils import *
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau


def test_class(cnn, valdataloader, lamda, gpuFlag):

    cnn.eval()
    test_labelloss = []
    test_closs = []
    accuracy = 0
    total = 0
    iBatch =0
    for x_batch, y_batch in valdataloader:

        if gpuFlag:
            x_batch = x_batch.cuda(0)
            y_batch = y_batch.cuda(0)

        y_pred = cnn.forward(x_batch.float())
        loss = cnn.loss(y_batch,lamda)

        if gpuFlag:
            y_batch = y_batch.cpu()
            y_pred = y_pred.cpu()

        accuracy += torch.sum(torch.eq(y_pred, y_batch)).item()
        total += len(y_batch)
        test_labelloss.extend([cnn.labelloss.item()] * x_batch.size()[0])
        test_closs.extend([cnn.closs.item()] * x_batch.size()[0])

        del x_batch
        del y_batch
        iBatch += 1

    return np.mean(test_labelloss), np.mean(test_closs), accuracy / total


def train(config):

    #_______ gpu configuration _____
    gpuAvaliable = torch.cuda.is_available()
    gpuFlag = gpuAvaliable and config.get('train', 'use_gpu') == "True"
    cudaID = None
    if gpuFlag:
        print("__ use gpu __")
        cudaID = int(config.get("train", "cuda_id"))

    #_______ model setting _____
    trainpath = config.get('path','data_path') + config.get('path','train_data')
    valpath = config.get('path', 'data_path') + config.get('path', 'val_data')

    batch_size = int(config.get('train','batch_size'))
    no_epoch = int(config.get('train','no_epoch'))

    lr = float(config.get('train','lr'))
    lr_cent = float(config.get('train','lr_cent'))
    weight_decay = float(config.get('train','weigth_decay'))
    momentum = float(config.get('train','momentum'))
    patience = int(config.get('train', 'patience'))

    lamda = float(config.get('model', 'lamda'))  # weight for center loss
    channels = list(map(int,config.get('model', 'channels').split(',')))
    blocks = list(map(int, config.get('model', 'blocks').split(',')))
    linears = list(map(int,config.get('model', 'linears').split(',')))
    feat_dim = int(config.get('model', 'feat_dim'))
    flatten = config.get('model', 'flatten') == 'True'

    with open('savedModel/label_dict.pickle', 'rb') as handle:
        b = pickle.load(handle)

    target_dict = b['label2id']

    traindataloader, num_class, train_size = load_data_val(trainpath, batch_size, target_dict)
    valdataloader, num_class, val_size = load_data_val(valpath, batch_size, target_dict)

    # a = {}
    # a['label2id'] = target_dict
    # a['id2label'] = label_dict
    # a['num_class'] = num_class
    # with open('savedModel/label_dict.pickle', 'wb') as handle:
    #     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    no_batch = int(train_size/batch_size)

    cnn = CNN(channels, blocks, linears, num_class, flatten, feat_dim, gpuFlag)
    if config.get('train', 'use_saved') == 'True':
        path = config.get('path', 'savedModelpath') + config.get('train', 'model_path')
        cnn.load_state_dict(torch.load(path))
        print("__ load saved model __")
    else:
        cnn.apply(init_xavier)
    closs = cnn.closs_criterion


    if gpuFlag:
        print("__ move models to GPU __")
        cnn = cnn.cuda(cudaID)
        closs = closs.cuda(cudaID)

    #loosce =  nn.CrossEntropyLoss()
    #closs = CenterLoss(num_class, feat_dim, gpuFlag)
    optim_label = torch.optim.SGD(cnn.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    optim_closs = torch.optim.SGD(closs.parameters(), lr=lr_cent)
    scheduler_label = ReduceLROnPlateau(optim_label, 'min',factor=0.1, patience=patience)
    scheduler_closs = ReduceLROnPlateau(optim_closs, 'min',factor=0.1, patience=patience)

    #_______ train _____
    for iEpoch in range(no_epoch):

        time_start = time.time()
        cnn.train()
        iBatch = 0
        avg_loss = 0.0
        avg_acc = 0.0

        for x_batch, y_batch in traindataloader:

            if gpuFlag:
                x_batch = x_batch.cuda(cudaID)
                y_batch = y_batch.cuda(cudaID)

            optim_label.zero_grad()
            optim_closs.zero_grad()
            y_pred = cnn.forward(x_batch.float()) # label output

            loss = cnn.loss(y_batch, lamda)
            loss.backward()

            optim_label.step()
            # by doing so, weight_cent would not impact on the learning of centers
            for param in closs.parameters():
                param.grad.data *= (1. / lamda)
            optim_closs.step()

            avg_loss += loss.item()

            if gpuFlag:
                y_batch = y_batch.cpu()
                y_pred = y_pred.cpu()

            avg_acc += torch.sum(torch.eq(y_pred, y_batch)).item()/batch_size
            time_train = time.time()
            if iBatch % 100 == 0:
                print("loss: {:.4f}".format(loss),
                      "acc: {:.4f}".format(avg_acc/(iBatch+1)),
                      "epoch: ", iEpoch + 1, "/", no_epoch,
                      "batch: ", iBatch + 1, "/", no_batch,
                      "training time: {:.2f}".format(time_train - time_start))

            iBatch += 1
            
        val_celoss, val_closs, val_acc = test_class(cnn,valdataloader, lamda, gpuFlag)
        #train_loss, train_acc = test_classify_closs(model, data_loader)
        scheduler_label.step(val_celoss)
        scheduler_closs.step(val_closs)

        print('Val Loss: {:.4f}\tVal Accuracy: {:.4f}'.format(val_celoss, val_acc))
        bestmodel = saveModel(cnn.state_dict(), iEpoch)

    print(bestmodel)
    return

if __name__ == '__main__':

    savedModelPath = './savedModel/'
    if not os.path.isdir(savedModelPath):
        os.makedirs(savedModelPath)

    configPath = "./config.ini"
    config = configparser.ConfigParser()
    config.read(configPath)

    train(config)