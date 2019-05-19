#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__doc__ = 'description:封装word2vec模型训练'
__author__ = '13314409603@163.com'

import sys
from pyltp import SentenceSplitter
import os
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences
from Config.utils import my_segmentor
from Config.config_parser import getParser


class Word2VecModel(object):
    def __init__(self,model_save_path,train_data_save_path,size,window=5,min_count=1,workers=4):
        self.save_path = os.path.join(model_save_path,'word2vec.model')
        self.size = size
        self.min_count = min_count
        self.window = window
        self.workers = workers
        if(os.path.exists(self.save_path)):
            self.model = Word2Vec.load(self.save_path)
        else:
            print('word2vec model first train')
            # sentencs =
            self.model = Word2Vec(PathLineSentences(train_data_save_path),size=self.size,window=self.window,min_count=self.min_count,workers=self.workers,
                                  sg=1)#1-skip,0-cbow
            print('word2vec model first train end!')

            self.save()
            print('word2vec model save :'+self.save_path)

    def reTrain(self,train_data_save_path):
        # self.model.build_vocab(MySentences(train_data_save_path)
        total_examples = 0
        for file in os.listdir(train_data_save_path):
            with open(os.path.join(train_data_save_path,file),'r',encoding='utf8')as f:
                total_examples += len(f.readlines())
        self.model.train(PathLineSentences(train_data_save_path),total_examples=total_examples,epochs=5)
        print('word2vec retrain end!')

    def save(self):
        self.model.save(self.save_path)

    def getEmbedded(self):
        return self.model.wv

#将训练数据分词，用作训练word2vec
#source-训练数据源地址，文件夹下包含cpws,qstsbl,qsz
#savePath-分词之后单个文件存放位置
#segmentor_model_path-分词模型存放位置
#segmentor_user_dict_path-自定义分词词典位置
#stop_words_path-停用词位置
def segment_words(source,savePath,stop_words_path):
    #分词器
    segmentor = my_segmentor

    if ( not os.path.exists(savePath)):
        os.mkdir(savePath)

    #停用词
    with open(stop_words_path, 'r', encoding='utf8') as f:
        stopWords = set(f.read().split())

    def handFile(path,prefx):
        if(os.path.isdir(path)):
            for fileNmae in os.listdir(path):
                handFile(os.path.join(path,fileNmae),prefx)
        else:
            handleSingleFile(path,prefx)

    def handleSingleFile(path,prefx):
        with open(path,'r',encoding='utf8') as f,open(os.path.join(savePath, prefx+'_'+os.path.basename(path)), 'w', encoding='utf8') as fw:
            content = f.read()
            sentences = SentenceSplitter.split(content)
            for str in sentences:
                # 分词
                words = segmentor.segment(str)

                # 去停用词
                words = list(filter(lambda x:False if(x in stopWords) else True,words))
                if (len(words) == 0):
                    continue

                fw.write(' '.join(list(words)))
                fw.write('\n')

    #分别处理三类文件夹
    qstsbl = os.path.join(source, 'qstsbl')
    handFile(qstsbl, 'qstsbl')
    qsz = os.path.join(source, 'qsz')
    handFile(qsz, 'qsz')
    cpws = os.path.join(source,'cpws')
    handFile(cpws,'cpws')


if __name__ == '__main__':
    FLAGES = getParser()
    dim = FLAGES.embedded_dim
    rootdir = FLAGES.root_dir
    ltpDir = os.path.join(rootdir,'ltp_data_v3.4.0')
    word2vec_dir = os.path.join(rootdir,'word2vec')

    word2vec_train_dir = os.path.join(word2vec_dir,'train')
    if (not os.path.exists(word2vec_train_dir) ):
        os.mkdir(word2vec_train_dir)

    #分词
    segment_words(os.path.join(rootdir,'原始_待分句_样例'),
                  word2vec_train_dir,
                  os.path.join(rootdir,'newStopWords.txt'))
    #训练或者加载已有模型
    wv = Word2VecModel(word2vec_dir, word2vec_train_dir, dim)
    wv = wv.getEmbedded()
    print(wv.most_similar('原告'))
    print(wv.similarity('原告', '被告'))
    print(wv['原告'])
    print(wv['被告'])
    sys.exit(0)