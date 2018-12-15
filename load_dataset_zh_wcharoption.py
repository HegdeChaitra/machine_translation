import pandas as pd
import numpy as np
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
import pickle
import random
import pdb
from torch.utils.data import DataLoader

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
    def __init__(self, name, minimum_count = 3):
        self.name = name
        self.word2index = {}
        self.word2count = {}
#         self.index2word = {0: "SOS", 1: "EOS", 2:"UKN",3:"PAD"}
        self.index2word = ["SOS","EOS","UKN","PAD"]
        self.n_words = 4  # Count SOS and EOS
        self.minimum_count = minimum_count
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word.lower())
#             if word not in string.punctuation:
#                 self.addWord(word.lower())
    def addSentence_zh(self, sentence):
        for word in list(sentence):
            self.addWord(word.lower())

    def addWord(self, word):
        if word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1
        if self.word2count[word] >= self.minimum_count:
            if word not in self.word2index:
                self.word2index[word] = self.n_words
    #             self.index2word[self.n_words] = word
                self.index2word.append(word)
                self.n_words += 1
            
            
            
def split(df, char=False):
    df['en_tokenized'] = df["en_data"].apply(lambda x:x.split( ))
    if char:
        df['zh_tokenized'] = df['zh_data'].apply(lambda x:list(x))
    else:
        df['zh_tokenized'] = df['zh_data'].apply(lambda x:x.split())
    return df



def token2index_dataset(df,en_lang,zh_lang):
    for lan in ['en','zh']:
        indices_data = []
        if lan=='en':
            lang_obj = en_lang
        else:
            lang_obj = zh_lang
        for tokens in df[lan+'_tokenized']:
            index_list = [lang_obj.word2index[token.lower()] if token.lower() in lang_obj.word2index else UNK_IDX for token in tokens]
            index_list.append(EOS_token)
#             index_list.insert(0,SOS_token)
            indices_data.append(index_list)
        df[lan+'_idized'] = indices_data
    return df


def train_val_load(MAX_LEN, old_lang_obj, path, char=False):
    if char:
        en_train = read_dataset(path+"/iwslt-zh-en-processed/train.en")
        en_val = read_dataset(path+"/iwslt-zh-en-processed/dev.en")
        
        zh_train = read_dataset(path+"/iwslt-zh-en-processed/train.zh")
        zh_val = read_dataset(path+"/iwslt-zh-en-processed/dev.zh")
    else:
        en_train = read_dataset(path+"/iwslt-zh-en/train.tok.en")
        en_val = read_dataset(path+"/iwslt-zh-en/dev.tok.en")
        
        zh_train = read_dataset(path+"/iwslt-zh-en/train.tok.zh")
        zh_val = read_dataset(path+"/iwslt-zh-en/dev.tok.zh")
    
    train = pd.DataFrame()
    train['en_data'] = en_train['data']
    train['zh_data'] = zh_train['data']
    
    val = pd.DataFrame()
    val['en_data'] = en_val['data']
    val['zh_data'] = zh_val['data']
    
    if old_lang_obj:
        with open(old_lang_obj,'rb') as f:
            en_lang = pickle.load(f)
            zh_lang = pickle.load(f)
    else:
        en_lang = Lang("en")
        for ex in train['en_data']:
            en_lang.addSentence(ex)

        if char:
            zh_lang = Lang("zh")
            for ex in train['vi_data']:
                zh_lang.addSentence_zh(ex)
        else:
            zh_lang = Lang("zh")
            for ex in train['zh_data']:
                zh_lang.addSentence(ex)
        
    train = split(train, char=char)
    val = split(val, char=char)
    
    train = token2index_dataset(train,en_lang,zh_lang)
    val = token2index_dataset(val,en_lang,zh_lang)
    
    train['en_len'] = train['en_idized'].apply(lambda x: len(x))
    train['zh_len'] = train['zh_idized'].apply(lambda x:len(x))
    
    val['en_len'] = val['en_idized'].apply(lambda x: len(x))
    val['zh_len'] = val['zh_idized'].apply(lambda x: len(x))
    
    train = train[np.logical_and(train['en_len']>=2,train['zh_len']>=2)]
#     train = train[train['vi_len']<=MAX_LEN]
    
    val = val[np.logical_and(val['en_len']>=2,val['zh_len']>=2)]
#     val = val[val['vi_len']<=MAX_LEN]
    
    return train,val,en_lang,zh_lang