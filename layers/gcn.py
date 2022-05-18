import copy
import json
import math
import six
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class gcnlayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self,  mem_dim, layers, droput=0.5):
        super(gcnlayer, self).__init__()
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(droput)

        # linear transformation
        self.linear_output = nn.Linear(self.mem_dim, self.mem_dim)

        # dcgcn block
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.linear_output = self.linear_output.cuda()

    def forward(self, gcn_inputs, adj):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1

        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):

            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            AxW = AxW + self.weight_list[l](outputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))

        gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = gcn_outputs + gcn_inputs

        out = self.linear_output(gcn_outputs)

        return out, adj

if __name__ =="__main__":
    gcn = gcnlayer(200,2).cuda()
    input = torch.randn(8,100,200).cuda()
    adj = torch.randn(8,100,100).cuda()
    out = gcn(input,adj)
    print(out[0].shape)