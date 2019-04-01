#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__doc__ = 'description'
__author__ = '13314409603@163.com'

import sys
from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#句子生成器
class SentenceGetter(object):
    def __init__(self,data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s:[(w,p,t) for w,p,t in zip(s['Word'].values.tolist(),s['POS'].values.tolist(),s['Tag'].values.tolist())]

        self.grouped = self.data.groupby('Sentence #').apply(agg_func)
        self.sentences = [s for s in self.grouped]
    def get_next(self):
        try:
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

#基于记忆的预测器
class MemoryTagger(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.tags = set()
        self.memory = {}
    #训练
    def fit(self,X,y):
        #输入一串word X和tag y
        voc = {} #存储每个词个每个tag次数
        for x,t in zip(X,y):
            self.tags.add(t)
            if x in voc:
                if(t in voc[x]):
                    voc[x][t] += 1
                else:
                    voc[x][t] = 1
            else:
                voc[x] = {t:1}
        for k,d in voc.items():
            self.memory[k] = max(d,key=d.get) #max 针对字典会返回最大键值，后面key跟针对第一个参数的函数，这里就会针对键值

    def predict(self,X,y=None):
        #预测
        return [self.memory.get(x,'O') for x in X]

def main():
    data = pd.read_csv('ner_dataset.csv',encoding='latin1')

    data = data.fillna(method="ffill")

    words = list(set(data['Word'].values))
    words.append('ENDPAD')
    n_words = len(words)

    tags = list(set(data['Tag'].values))
    n_tages = len(tags)

    getter = SentenceGetter(data)
    sent = getter.get_next()
    print(sent)
    sentences = SentenceGetter.sentences

    max_len = 75
    word2idx = {w:i+1 for i,w in enumerate(words)}
    tag2idx = {t:i for i,t in enumerate(tags)}
    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X = pad_sequences(maxlen=max_len,sequences=X,padding='POST',value=n_words-1)

    y = [[tag2idx[w[2]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=max_len,sequences=y,padding='post',value=tag2idx['O'])
    y = [to_categorical(i,num_classes=n_tages) for i in y]

    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.1)

    from keras.models import Model,Input
    from keras.layers import LSTM,Embedding,Dense,TimeDistributed,Dropout,Bidirectional
    # from

if __name__ == '__main__':
    main()
    sys.exit(0)