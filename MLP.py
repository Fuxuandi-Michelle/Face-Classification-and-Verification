import torch
import torch.nn as nn
from loss import CenterLoss

class MLP(nn.Module):
    def __init__(self, input_dim, layers, feat_dim, num_class, gpu_flag = True):
        super(MLP, self).__init__()
        self.CE = torch.nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.input_dim = input_dim
        self.layers = [self.input_dim]+layers+[num_class]
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.emb_linears = nn.Linear(self.layers[0], self.layers[1], bias=True)

        self.label_linears = []
        for i in range(1, len(self.layers) - 1):
            self.label_linears.append(nn.BatchNorm1d(self.layers[i]))
            self.label_linears.append(nn.PReLU())
            self.label_linears.append(nn.Linear(self.layers[i], self.layers[i+1], bias=True))

        #self.linears.append(nn.Linear(self.layers[-2], self.layers[-1], bias=True))
        self.label_linears = nn.Sequential(*self.label_linears)

        self.closs_criterion = CenterLoss(self.num_class, self.feat_dim, use_gpu=gpu_flag)
        self.ce_criterion = nn.CrossEntropyLoss()

    def forward(self,input, emb):
        #input: [BSZ T 40]
        #input = torch.flatten(input, start_dim=1)
        #img = img.flatten(start_dim=1)
        input = torch.cat((input,emb), dim=1)
        self.emb_output = self.emb_linears(input)
        self.label_output = self.label_linears(self.emb_output)
        #self.prob = self.softmax(self.output) #[BSZ 138]
        self.pred = torch.argmax(self.label_output, dim=1)  #[BSZ]
        return self.pred

    def loss(self,target,lamda):
        self.labelloss = self.ce_criterion(self.label_output, target)
        self.closs = self.closs_criterion(self.emb_output, target)
        return self.labelloss + lamda * self.closs
