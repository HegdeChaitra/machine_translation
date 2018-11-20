import pandas as pd
import numpy as np
import unicodedata
import string
import re
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
import pickle as pkl
import random
import pdb
from torch.utils.data import DataLoader

from io import open
import unicodedata
import string
import re
import random

from torch import optim
import time

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size,bi):
        super(EncoderRNN, self).__init__()
        self.bi=bi
        if self.bi:
            self.mul=2
        else:
            self.mul=1
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size,batch_first=True,bidirectional=self.bi)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self,bs):
        return torch.zeros(self.mul, bs, self.hidden_size).cuda()
    
    
class AttentionDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size,bi, MAX_LEN):
        super(AttentionDecoderRNN, self).__init__()
        self.bi = bi
        if self.bi:
            self.mul=2
        else:
            self.mul=1
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(p=0.1)
        self.gru = nn.GRU(hidden_size, hidden_size,batch_first=True,bidirectional=self.bi)
        
        self.attn = nn.Linear(self.hidden_size * self.mul, MAX_LEN)
        self.attn_combine = nn.Linear(self.hidden_size * self.mul+self.hidden_size, self.hidden_size)
        
        self.out = nn.Linear(self.mul*hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden,encoder_outputs):
        bss = input.size(0)
        output = self.embedding(input)
        output = self.dropout(output)

        cat = torch.cat((output, hidden[0].unsqueeze(1)), 2)
        att_out = F.softmax(self.attn(cat),dim=1)
        attn_applied = torch.bmm(att_out,encoder_outputs)
        attn_cat = torch.cat((output, attn_applied), 2)
        attn_comb = self.attn_combine(attn_cat)
        
        output = F.relu(attn_comb)
        output, hidden = self.gru(output, hidden)
        output = self.out(output.squeeze(dim=1))
        output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        return torch.zeros(self.mul, bs, self.hidden_size).cuda()   
    
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size,bi):
        super(DecoderRNN, self).__init__()
        self.bi = bi
        if self.bi:
            self.mul=2
        else:
            self.mul=1
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size,batch_first=True,bidirectional=self.bi)
        self.out = nn.Linear(self.mul * hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output).squeeze(dim=1))

        return output, hidden

    def initHidden(self):
        return torch.zeros(self.mul, bs, self.hidden_size).cuda()
    
