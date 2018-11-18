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

import unicodedata
import string
import re
import random
import argparse
from torch import optim
import time

UNK_IDX = 2
PAD_IDX = 3
SOS_token = 0
EOS_token = 1


def encode_decode(encoder,decoder,data_en,data_de,max_len,m_type):
    use_teacher_forcing = True if random.random() < 0.5 else False
    bss = data_en.size(0)
    en_h = encoder.initHidden(bss)
    en_out,en_hid = encoder(data_en,en_h)
    
    decoder_hidden = en_hid
    decoder_input = torch.tensor([[SOS_token]]*bss).cuda()

    if use_teacher_forcing:
        d_out = []
        for i in range(max_len):
            if m_type=="attention":
                decoder_output,decoder_hidden = decoder(decoder_input,decoder_hidden,en_out)
            else:
                decoder_output,decoder_hidden = decoder(decoder_input,decoder_hidden)
            d_out.append(decoder_output)
            decoder_input = data_de[:,i].view(-1,1)
        d_hid = decoder_hidden
        d_out = torch.cat(d_out,dim=0)
    else:
        d_out = []
        for i in range(max_len):
            if m_type=="attention":
                decoder_output,decoder_hidden = decoder(decoder_input,decoder_hidden,en_out)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input,decoder_hidden)
            d_out.append(decoder_output)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach().view(-1,1)
        d_hid = decoder_hidden
        d_out = torch.cat(d_out,dim=0)
    return d_out, d_hid


def train_model(encoder_optimizer,decoder_optimizer, encoder, decoder, loss_fun,max_len, m_type, dataloader, num_epochs=60):
    best_score = 0
    best_au = 0
    loss_hist = {'train': [], 'validate': []}
    acc_hist = {'train': [], 'validate': []}
    for epoch in range(num_epochs):
        for ex, phase in enumerate(['train', 'validate']):
            start = time.time()
            total = 0
            top1_correct = 0
            running_loss = 0
            running_total = 0
            if phase == 'train':
                encoder.train(True)
                decoder.train(True)
            else:
                encoder.train(False)
                decoder.train(False)
            for data in dataloader[phase]:
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                encoder_i = data[0].cuda()
                decoder_i = data[1].cuda()
                                
                out, hidden = encode_decode(encoder,decoder,encoder_i,decoder_i,max_len,m_type)
                loss = loss_fun(out.float(), decoder_i.long().view(-1))
                N = decoder_i.size(0)
                running_loss += loss.item() * N
                
                total += N
                if phase == 'train':
                    loss.backward()
                    encoder_optimizer.step()
                    decoder_optimizer.step()

            epoch_loss = running_loss / total
            epoch_acc = 0
            loss_hist[phase].append(epoch_loss)
            acc_hist[phase].append(epoch_acc)
            print("epoch {} {} loss = {}, accurancy = {} time = {}".format(epoch, phase, epoch_loss, epoch_acc,
                                                                           time.time() - start))
        if phase == 'validate' and epoch_acc > best_score:
            best_score = epoch_acc
    print("Training completed. Best accuracy is {}".format(best_score))
    return encoder,decoder,loss_hist,acc_hist



