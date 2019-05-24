#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os

__doc__ = 'description'
__author__ = '13314409603@163.com'

def getParser():
    # 构造参数
    rootPath = 'C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\'
    # rootPath = '/root/lstm_crf/data'
    ltpPath = os.path.join(rootPath, 'ltp_data_v3.4.0')
    parser = argparse.ArgumentParser(description='Bi-LSTM+CRF')
    parser.add_argument('--root_dir', help='root dir', default=rootPath)
    parser.add_argument('--ifTrain', help='train and dev', default=False)
    parser.add_argument('--ifTest', help='test', default=False)
    parser.add_argument('--ifPredict',help='predict input sentence',default=False)
    parser.add_argument('--ifPredictFile',help='predict sentence in file or dir',default=False)
    parser.add_argument('--dropout_rate', help='dropout rate', default=0.9)
    parser.add_argument('--learning_rate', help='learning rate', default=0.001)
    parser.add_argument('--hidden_units', help='hidden units', default=100)
    parser.add_argument('--num_layers', help='num of layers', default=1)
    parser.add_argument('--sentence_mode',
                        # help='one sentence one event is Spe,one sentence may have many events is Full,or merge', default='Full')
                        # help='one sentence one event is Spe,one sentence may have many events is Full,or merge', default='Spe')
                        help='one sentence one event is Spe,one sentence may have many events is Full,or merge', default='Merge')
    parser.add_argument('--labeled_data_path', help='labeled data path',
                        # default=os.path.join(os.path.join(rootPath, 'labeled'), 'Merge_for_first'))
                        default=os.path.join(os.path.join(rootPath, 'labeled'), 'Merge_for_second'))
    # parser.add_argument('--max_sequence_length', help='max length of sequence', default= 40 )  # Full - 40,Spe-51,Merge-55
    # parser.add_argument('--max_sequence_length', help='max length of sequence', default= 51 )  # Full - 40,Spe-51,Merge-55
    parser.add_argument('--max_sequence_length', help='max length of sequence', default= 55 )  # Full - 40,Spe-51,Merge-55
    parser.add_argument('--batch_size', help='batch size', default=64)
    parser.add_argument('--num_epochs', help='num of epochs', default=10)
    parser.add_argument('--device_map', help='which device to see', default='CPU:0')
    parser.add_argument('--segmentor_model_path', help='segmentor model path',
                        default=os.path.join(ltpPath, 'cws.model'))
    parser.add_argument('--segmentor_user_dict_path', help='segmentor user dictionary path',
                        default=os.path.join(ltpPath, 'userDict.txt'))
    parser.add_argument('--word2vec_path', help='word2vecpath', default=os.path.join(rootPath, 'word2vec'))
    parser.add_argument('--embedded_dim', help='wordembeddeddim', default=300)
    FLAGS, args = parser.parse_known_args()
    return FLAGS ;

if __name__ == '__main__':
    pass