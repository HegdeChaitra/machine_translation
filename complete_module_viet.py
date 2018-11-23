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
import pickle
import random
import pdb
from torch.utils.data import DataLoader
import logging

import unicodedata
import string
import re
import random
import argparse
from torch import optim
import time
import os
from validation import *
from bleu_score import BLEU_SCORE


from models_viet import EncoderRNN, AttentionDecoderRNN, DecoderRNN

from load_dataset_viet import *

from define_training_viet import *

parser = argparse.ArgumentParser()

parser.add_argument('--bs', type=int, default=32,
                    help='Batch size')

parser.add_argument('--num_epochs', type=int, default=50,
                    help='Number of epochs')

parser.add_argument('--save_dir', type=str, default='./saved_files/',
                    help='directory where models are to be saved. default=./saved_files/')

parser.add_argument('--model_name', type=str, required=True,
                    help='Name of model to be saved, required=True')

parser.add_argument('--data_path', type=str, default='.',
                    help="Path to data,  default='.' (current directory)")

parser.add_argument('--model_load', type=str, default=None,
                    help='model to load for continued training. default=None')

parser.add_argument('--max_len', type=int, default=48,
                    help='Maximum length of input sentence to consider. default=30')

parser.add_argument('--hid_size', type=int, default=100,
                    help='hidden size for GRU. default=100')

parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate. default=1e-4')

parser.add_argument('--bi', type=bool, default=True,
                    help='Use bidirectional GRUs or not. default=True')

parser.add_argument('--lang_obj', type=str, default="lang_obj.pkl",
                    help='Previously saved language objects. For the first run, lang_obj="", for 2nd onwards saved obj has to be specified. Default:lang_obj.pkl')

parser.add_argument('--type', type=str, default="no_attention",
                    help='attention model or non attention model. options=["attention","no_attention"]. default="no_attention"')

parser.add_argument('--att_type', type=str, default=None,
                    help='attention type. options=["type2", None]. default=None')

args = parser.parse_args()


class Vietnamese(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        english = self.df.iloc[idx]['en_idized']
        viet = self.df.iloc[idx]['vi_idized']
        en_len = self.df.iloc[idx]['en_len']
        vi_len = self.df.iloc[idx]['vi_len']
        return [english,viet,en_len,vi_len]
    
    
def vocab_collate_func(batch):
    en_data = []
    vi_data = []
    en_len = []
    vi_len = []

    for datum in batch:
        en_len.append(datum[2])
        vi_len.append(datum[3])
    # padding
    for datum in batch:
        if datum[2]>args.max_len:
            padded_vec_s1 = np.array(datum[0])[:MAX_LEN]
        else:
            padded_vec_s1 = np.pad(np.array(datum[0]),
                                pad_width=((0,MAX_LEN - datum[2])),
                                mode="constant", constant_values=PAD_IDX)
        if datum[3]>args.max_len:
            padded_vec_s2 = np.array(datum[1])[:MAX_LEN]
        else:
            padded_vec_s2 = np.pad(np.array(datum[1]),
                                pad_width=((0,MAX_LEN - datum[3])),
                                mode="constant", constant_values=PAD_IDX)
        en_data.append(padded_vec_s1)
        vi_data.append(padded_vec_s2)
        
    return [torch.from_numpy(np.array(vi_data)), torch.from_numpy(np.array(en_data)),
            torch.from_numpy(np.array(vi_len)), torch.from_numpy(np.array(en_len))]

if __name__=='__main__':
    MAX_LEN = args.max_len
    print("start")
    train,val,en_lang,vi_lang = train_val_load(args.max_len, args.lang_obj, args.data_path)
    train = train.sample(n=train.shape[0]//4)
    
    
    transformed_dataset = {'train': Vietnamese(train), 'validate': Vietnamese(val)}

    dataloader = {x: DataLoader(transformed_dataset[x], batch_size=args.bs, collate_fn = vocab_collate_func,
                                shuffle = True, num_workers=0) for x in ['train', 'validate']}
    
    print("data loader created")

    if args.model_load:
        encoder = torch.load(args.save_dir+args.model_load+"_enc")
        decoder = torch.load(args.save_dir+args.model_load+"_dec")
    else:
        encoder = EncoderRNN(vi_lang.n_words,args.hid_size,args.bi).cuda()
        if args.type=="attention":
            decoder = AttentionDecoderRNN(args.hid_size,en_lang.n_words,args.bi, MAX_LEN, attention_type = args.att_type).cuda()
        else:
            decoder = DecoderRNN(args.hid_size,en_lang.n_words,args.bi).cuda()
     
    print("encoder decoder objects created")
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    criterion = nn.NLLLoss()
    
    print("starting the training process")
    enc, dec, loss_hist, acc_hist = train_model(encoder_optimizer, decoder_optimizer, encoder, decoder, criterion,
                                               args.max_len, args.type, dataloader,en_lang, num_epochs = args.num_epochs)
    
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
        
    print("saving the models")
    torch.save(enc,args.save_dir+args.model_name+"_enc")
    torch.save(dec,args.save_dir+args.model_name+"_dec")
    
    with open(args.save_dir+args.model_name+"_history",'wb') as f:
        pickle.dump(loss_hist,f)
        pickle.dump(acc_hist,f)
    
    
    
    