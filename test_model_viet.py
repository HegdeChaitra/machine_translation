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

parser.add_argument('--save_dir', type=str, default='./saved_files/',
                    help='directory where models are to be saved. default=./saved_files/')

parser.add_argument('--data_path', type=str, default='.',
                    help="Path to data,  default='.' (current directory)")

parser.add_argument('--model_load', type=str, required=True,
                    help='model to load for continued training. required=True')

parser.add_argument('--max_len', type=int, default=30,
                    help='Maximum length of input sentence to consider. default=30')

parser.add_argument('--lang_obj', type=str, required=True,
                    help='Previously saved language objects, required=True')

parser.add_argument('--type', type=str, default="no_attention",
                    help='attention model or non attention model. options=["attention","no_attention"]. default="no_attention"')

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
    
    train,val,en_lang,vi_lang = train_val_load(args.max_len, args.lang_obj, args.data_path)
    
    transformed_dataset = {'train': Vietnamese(train), 'validate': Vietnamese(val)}

    dataloader = {x: DataLoader(transformed_dataset[x], batch_size=args.bs, collate_fn = vocab_collate_func,
                                shuffle = True, num_workers=0) for x in ['train', 'validate']}
    
    data = next(iter(dataloader['validate']))
    
    encoder = torch.load(args.save_dir+args.model_load+"_enc")
    decoder = torch.load(args.save_dir+args.model_load+"_dec")
    
    out,hid = encode_decode(encoder,decoder,data[0].cuda(),data[1].cuda(),args.max_len,args.type)
    _, top1_predicted = torch.max(out,dim = 1)

    for i in range(args.bs):
        print("-"*50)
        print("Given: "," ".join(np.array(en_lang.index2word)[data[1][i]][:data[3][i]]))
        print("Predicted: "," ".join(np.array(en_lang.index2word)[top1_predicted[i]]))
#         [:data[3][i]]
    
#     given=[]
#     for i in data[1][1]:
#         given.append(en_lang.index2word[i.item()])
        
#     pred = []
#     for i in top1_predicted[0]:
# #     print(i.item())
#         pred.append(en_lang.index2word[i.item()])

#     print("given: "," ".join(given))
#     print("predicted: "," ".join(pred))