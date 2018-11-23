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
from validation import *

UNK_IDX = 2
PAD_IDX = 3
SOS_token = 0
EOS_token = 1

from bleu_score import BLEU_SCORE


def convert_idx_2_sent(tensor, lang_obj):
    word_list = []
    for i in tensor:
        if i.item() not in set([PAD_IDX,EOS_token,SOS_token]):
            word_list.append(lang_obj.index2word[i.item()])
    return (' ').join(word_list)

def validation(encoder, decoder, dataloader, loss_fun, lang_en, max_len,m_type):
    encoder.train(False)
    decoder.train(False)
    pred_corpus = []
    true_corpus = []
    running_loss = 0
    running_total = 0
    bl = BLEU_SCORE()
    for data in dataloader:
        encoder_i = data[0].cuda()
        decoder_i = data[1].cuda()
        bs,sl = encoder_i.size()[:2]
        out, hidden = encode_decode(encoder,decoder,encoder_i,decoder_i,max_len,m_type, rand_num = 0)
        loss = loss_fun(out.float(), decoder_i.long())
        running_loss += loss.item() * bs
        running_total += bs
        pred = torch.max(out,dim = 1)[1]
        for t,p in zip(data[1],pred):
            t,p = convert_idx_2_sent(t,lang_en), convert_idx_2_sent(p,lang_en)
            true_corpus.append(t)
            pred_corpus.append(p)
    score = bl.corpus_bleu(pred_corpus,[true_corpus],lowercase=True)[0]
    return running_loss/running_total, score


def encode_decode(encoder,decoder,data_en,data_de,max_len,m_type, rand_num = 0.5):
    use_teacher_forcing = True if random.random() < rand_num else False
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
            d_out.append(decoder_output.unsqueeze(-1))
            decoder_input = data_de[:,i].view(-1,1)
        d_hid = decoder_hidden
        d_out = torch.cat(d_out,dim=-1)
    else:
        d_out = []
        for i in range(max_len):
            if m_type=="attention":
                decoder_output,decoder_hidden = decoder(decoder_input,decoder_hidden,en_out)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input,decoder_hidden)
            d_out.append(decoder_output.unsqueeze(-1))
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach().view(-1,1)
        d_hid = decoder_hidden
        d_out = torch.cat(d_out,dim=-1)
    return d_out, d_hid


def train_model(encoder_optimizer,decoder_optimizer, encoder, decoder, loss_fun,max_len, m_type, dataloader, en_lang,\
                num_epochs=60, val_every = 1, train_bleu_every = 10):
    best_score = 0
    best_bleu = 0
    loss_hist = {'train': [], 'validate': []}
    bleu_hist = {'train': [], 'validate': []}
    best_encoder_wts = None
    best_decoder_wts = None
    for epoch in range(num_epochs):
        for ex, phase in enumerate(['train']):
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
                loss = loss_fun(out.float(), decoder_i.long())
                N = decoder_i.size(0)
                running_loss += loss.item() * N
                
                total += N
                if phase == 'train':
                    loss.backward()
                    encoder_optimizer.step()
                    decoder_optimizer.step()
            epoch_loss = running_loss / total
            loss_hist[phase].append(epoch_loss)
            print("epoch {} {} loss = {}, time = {}".format(epoch, phase, epoch_loss,
                                                                           time.time() - start))
#             if epoch%train_bleu_every ==0:
#                 train_loss, train_bleu_score = validation(encoder,decoder, dataloader['train'],loss_fun, en_lang,max_len,m_type)
#                 bleu_hist['train'].append(train_bleu_score)
#                 print("Train BLEU = ", train_bleu_score)
            if epoch%val_every == 0:
                val_loss, val_bleu_score = validation(encoder,decoder, dataloader['validate'],loss_fun, en_lang ,max_len,m_type)
                loss_hist['validate'].append(val_loss)
                bleu_hist['validate'].append(val_bleu_score)
                print("validation loss = ", val_loss)
                print("validation BLEU = ", val_bleu_score)
                if val_bleu_score > best_bleu:
                    best_bleu = val_bleu_score
                    best_encoder_wts = encoder.state_dict()
                    best_decoder_wts = decoder.state_dict()
            print('='*50)
    encoder.load_state_dict(best_encoder_wts)
    decoder.load_state_dict(best_decoder_wts)
    print("Training completed. Best BLEU is {}".format(best_bleu))
    return encoder,decoder,loss_hist,bleu_hist



