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

from torch import optim
import time

UNK_IDX = 2
PAD_IDX = 3
SOS_token = 0
EOS_token = 1

def read_dataset(file):
    f = open(file)
    list_l = []
    for line in f:
        list_l.append(line.strip())
    df = pd.DataFrame()
    df['data'] = list_l
    return df

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2:"UKN",3:"PAD"}
        self.n_words = 4  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word.lower())

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
            
def split(df):
    df['en_tokenized'] = df["en_data"].apply(lambda x:x.lower().split( ))
    df['vi_tokenized'] = df['vi_data'].apply(lambda x:x.lower().split( ))
    return df



def token2index_dataset(df,en_lang,vi_lang):
    for lan in ['en','vi']:
        indices_data = []
        if lan=='en':
            lang_obj = en_lang
        else:
            lang_obj = vi_lang
        for tokens in df[lan+'_tokenized']:
            index_list = [lang_obj.word2index[token] if token in lang_obj.word2index else UNK_IDX for token in tokens]
            index_list.append(EOS_token)
#             index_list.insert(0,SOS_token)
            indices_data.append(index_list)
        df[lan+'_idized'] = indices_data
    return df


def train_val_load(MAX_LEN, path):
    en_train = read_dataset(path+"/iwslt-vi-en/train.tok.en")
    en_val = read_dataset(path+"/iwslt-vi-en/dev.tok.en")
    
    vi_train = read_dataset(path+"/iwslt-vi-en/train.tok.vi")
    vi_val = read_dataset(path+"/iwslt-vi-en/dev.tok.vi")
    
    train = pd.DataFrame()
    train['en_data'] = en_train['data']
    train['vi_data'] = vi_train['data']
    
    val = pd.DataFrame()
    val['en_data'] = en_val['data']
    val['vi_data'] = vi_val['data']
    
    en_lang = Lang("en")
    for ex in train['en_data']:
        en_lang.addSentence(ex)
    
    vi_lang = Lang("vi")
    for ex in train['vi_data']:
        vi_lang.addSentence(ex)
        
    train = split(train)
    val = split(val)
    
    train = token2index_dataset(train,en_lang,vi_lang)
    val = token2index_dataset(val,en_lang,vi_lang)
    
    train['en_len'] = train['en_idized'].apply(lambda x: len(x))
    train['vi_len'] = train['vi_idized'].apply(lambda x:len(x))
    
    val['en_len'] = val['en_idized'].apply(lambda x: len(x))
    val['vi_len'] = val['vi_idized'].apply(lambda x: len(x))
    
    train = train[np.logical_and(train['en_len']>=2,train['vi_len']>=2)]
    train = train[train['vi_len']<=MAX_LEN]
    
    val = val[np.logical_and(val['en_len']>=2,val['vi_len']>=2)]
    val = val[val['vi_len']<=MAX_LEN]
    
    return train,val,en_lang,vi_lang