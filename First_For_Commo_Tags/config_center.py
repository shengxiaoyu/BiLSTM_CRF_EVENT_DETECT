#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description,lstm_crf模型的配置中心,包括' \
          'WV-word2vec模型,' \
          'TAG_2_ID,ID_2_TAG' \
          'pos标注模型POSTAGGER,' \
          '分词模型SEGMENTOR,' \
          '停用词集STOP_WORD' \
          '触发词字典TRIGGER_WORDS_DICT' \
          '触发词tag集TRIGGER_TAGs' \
          '参数tag集ARGU_TAGs'
__author__ = '13314409603@163.com'
import os
import numpy as np
import pandas as pd
from pyltp import Postagger
from pyltp import Segmentor
from Word2Vec.my_word2vec import Word2VecModel

WV = None
TRIGGER_TAGs = None
ARGU_TAGs =None
TAG_2_ID = None
ID_2_TAG = None
TAGs_LEN = 0
POSTAGGER = None
POS_2_ID = None
POSs_LEN = 0
# 分词器
SEGMENTOR = None
STOP_WORDS=None
TRIGGER_WORDS_DICT = None

ifInited = False

#初始化各类模型以及词集
def init(rootdir):
    global ifInited
    if(not ifInited):
        initTags(os.path.join(rootdir,'full_trigger_labels.txt'),os.path.join(rootdir, 'full_argu_labels.txt'))
        initPyltpModel(os.path.join(rootdir,'ltp_data_v3.4.0'))
        initWord2Vec(os.path.join(rootdir, 'word2vec'))
        initPosTag(os.path.join(rootdir, 'pos_tags.csv'))
        initStopWords(os.path.join(rootdir, 'newStopWords.txt'))
        initTriggerWords(os.path.join(rootdir,'triggers'))
        ifInited = True

def initTags(triggerLablePath,argumentLabelPath):
    global TAG_2_ID, ID_2_TAG,TAGs_LEN,TRIGGER_TAGs,ARGU_TAGs
    TAG_2_ID={}
    ID_2_TAG={}
    TRIGGER_TAGs=[]
    ARGU_TAGs = []
    # 把<pad>也加入tag字典
    TAG_2_ID['<pad>'] = 0
    ID_2_TAG[0] = '<pad>'

    TAG_2_ID['O'] = 1
    ID_2_TAG[1] ='O'
    # 读取根目录下的labelds文件生成tag—id
    index = 2
    # 获取触发词tag
    with open(triggerLablePath, 'r', encoding='utf8') as f:
        for line in f.readlines():
            TAG_2_ID[line.strip()] = index
            ID_2_TAG[index] = line.strip()
            TRIGGER_TAGs.append(line.strip())
            index += 1
    #获取参数tag
    with open(argumentLabelPath, 'r', encoding='utf8') as f:
        for line in f.readlines():
            TAG_2_ID[line.strip()] = index
            ID_2_TAG[index] = line.strip()
            ARGU_TAGs.append(line.strip())
            index += 1
    TAGs_LEN = len(TAG_2_ID)
def initWord2Vec(word2vec_model_path):
    global WV
    WV = Word2VecModel(word2vec_model_path, '', 30).getEmbedded()
    # <pad> -- <pad> fill word2vec and tags，添加一个<pad>-向量为0的，用于填充
    WV.add('<pad>', np.zeros(WV.vector_size))
def initPosTag(pos_tag_path):
    global POS_2_ID,POSs_LEN
    POS_2_ID={}
    posDict = pd.read_csv(pos_tag_path)
    for id,pos in zip(posDict['Index'],posDict['Tag']):
        POS_2_ID[pos]=id
    # POS_2_ID['<pad>'] = 0
    POSs_LEN = len(POS_2_ID)
def initPyltpModel(ltp_path):
    global POSTAGGER,SEGMENTOR
    POSTAGGER = Postagger()
    SEGMENTOR = Segmentor()
    #初始化词性标注模型
    POSTAGGER.load(os.path.join(ltp_path,'pos.model'))
    SEGMENTOR.load_with_lexicon(os.path.join(ltp_path,'cws.model'), os.path.join(ltp_path,'userDict.txt'))
def initStopWords(path):
    # 停用词集
    global STOP_WORDS
    with open(path, 'r', encoding='utf8') as f:
        STOP_WORDS = set(f.read().split())
def initTriggerWords(path):
    global TRIGGER_WORDS_DICT
    TRIGGER_WORDS_DICT = {}
    # 初始化触发词集
    for triggerFile in os.listdir(path):
        with open(os.path.join(path, triggerFile), 'r', encoding='utf8') as f:
            content = f.read()
        TRIGGER_WORDS_DICT[triggerFile.split('.')[0]] = set(filter(lambda x:False if(x=='') else True,content.split('\n')))
#释放模型
def release():
    global POSTAGGER, SEGMENTOR
    POSTAGGER.release()
    SEGMENTOR.release()


if __name__ == '__main__':
    pass