#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__doc__ = 'description'
__author__ = '13314409603@163.com'


import os
import argparse
import sys
from data_pre_process.qsz_process import QszProcess
from pyltp import Segmentor
from Word2Vec.my_word2vec import Word2VecModel
from LSTM_CRF.word2vec_lstm_crf_ed import main
import numpy as np

FLAGS = None

def processData():
    #分词模型
    segmentot = Segmentor()
    segmentot.load_with_lexicon(FLAGS.segmentor_model_path,FLAGS.segmentor_user_dict_path)
    qszProcess = QszProcess(FLAGS.source_dataset_path)
    qszProcess.segmentAndSave(segmentot,os.path.join(FLAGS.segment_result_save_path,'起诉状'))
    segmentot.release()

def trainWord2Vec(new_data_path=None):
    #训练word2vec，传入path表示使用新训练集继续训练
    wvm = Word2VecModel(FLAGS.word2vec_model_save_path,os.path.join(FLAGS.segment_result_save_path,'起诉状'),size=FLAGS.embedded_dim)
    if(new_data_path!=None):
        wvm.reTrain(new_data_path)
def testWore2vec():
    wv = Word2VecModel(FLAGS.word2vec_model_save_path, '',FLAGS.embedded_dim)
    wv = wv.getEmbedded()
    print(wv.most_similar('原告'))
    print(wv.similarity('原告', '被告'))
    print(wv['原告'])
    print(wv['被告'])

if __name__=='__main__':
    rootPath = 'C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\'
    parser = argparse.ArgumentParser(description='Bi-LSTM+CRF')
    parser.add_argument('--root_dir',help='root dir',default=rootPath)
    parser.add_argument('--source_dataset_path',help='source dataset path',default='E:\\研二1\\学术论文\\准备材料2\\离婚纠纷第二批（分庭审笔录）\\不含庭审笔录')
    parser.add_argument('--segment_result_save_path',help='save the Experimental data',default=os.path.join(rootPath,'segment_result'))
    parser.add_argument('--segmentor_model_path',help='segmentor model path',default=rootPath+'ltp_data_v3.4.0\\cws.model')
    parser.add_argument('--segmentor_user_dict_path',help='segmentor user dictionary path',default=rootPath+'ltp_data_v3.4.0\\userDict.txt')
    parser.add_argument('--word2vec_model_save_path',help='word2vec model save path',default=os.path.join(rootPath,'word2vec'))
    parser.add_argument('--embedded_dim',help='word embedded dim',default=30)
    parser.add_argument('--labeled_data_path',help='labeled data path',default=os.path.join(rootPath,'NERdata'))
    parser.add_argument('--batch_size',help='batch size',default=12)
    parser.add_argument('--num_epochs',help='num of epochs',default=10)
    parser.add_argument('--dropout_rate',help='dropout rate',default=0.9)
    parser.add_argument('--learning_rate',help='learning rate',default=0.001)
    parser.add_argument('--hidden_units',help='hidden units',default=100)
    parser.add_argument('--num_layers',help='num of layers',default=1)
    parser.add_argument('--max_sequence_length',help='max length of sequence',default=50)
    parser.add_argument('--mode',help='train or test or predict',default='train')
    parser.add_argument('--device_map',help='which device to see',default='CPU:0')
    FLAGS,args = parser.parse_known_args()
    # processData()
    # trainWord2Vec()
    # testWore2vec()
    main(FLAGS)
    sys.exit(0)