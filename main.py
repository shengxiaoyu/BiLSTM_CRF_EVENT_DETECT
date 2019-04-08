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
from data_pre_process.tsbl_process import TsblProcess
from data_pre_process.cpws_process import CpwsProcess
import LSTM_CRF.word2vec_lstm_crf_ed as run

FLAGS = None

#从原材料中生成分句后的起诉庭审笔录、起诉状、裁判文书
def processData():
    #分词模型
    segmentot = Segmentor()
    segmentot.load_with_lexicon(FLAGS.segmentor_model_path,FLAGS.segmentor_user_dict_path)

    qszProcess = QszProcess(FLAGS.source_dataset_path2)
    qszProcess.segmentAndSave(segmentot, os.path.join(FLAGS.segment_result_save_path, 'qstsbl'))
    #庭审笔录处理
    tsblProcess = TsblProcess(FLAGS.source_dataset_path2)
    tsblProcess.segmentAndSave(segmentot, os.path.join(FLAGS.segment_result_save_path, 'qstsbl'), 'paragraph')
    # #起诉状处理
    qszProcess = QszProcess(FLAGS.source_dataset_path)
    qszProcess.segmentAndSave(segmentot,os.path.join(FLAGS.segment_result_save_path,'qsz'))

    cpwsProcess = CpwsProcess(FLAGS.source_dataset_path3)
    cpwsProcess.segmentAndSave(segmentot,os.path.join(FLAGS.segment_result_save_path,'cpws'))

    segmentot.release()

def trainWord2Vec(new_data_path):
    #训练word2vec，传入path表示使用新训练集继续训练
    wvm = Word2VecModel(FLAGS.word2vec_path,os.path.join(FLAGS.word2vec_path,'train'),size=FLAGS.embedded_dim)
    if(new_data_path!=None):
        wvm.reTrain(new_data_path)
def testWore2vec():
    wv = Word2VecModel(FLAGS.word2vec_path, '',FLAGS.embedded_dim)
    wv = wv.getEmbedded()
    print(wv.most_similar('原告'))
    print(wv.similarity('原告', '被告'))
    print(wv.most_similar('结婚'))
    print(wv.similarity('结婚', '离婚'))
    print(wv.most_similar('争吵'))
    print(wv.most_similar('洗衣机'))
    print(wv.most_similar('夫妻'))
    print(wv.most_similar('生育'))
    print(wv.most_similar('诉讼'))

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
    parser.add_argument('--dropout_rate', help='dropout rate', default=0.9)
    parser.add_argument('--learning_rate', help='learning rate', default=0.001)
    parser.add_argument('--hidden_units', help='hidden units', default=100)
    parser.add_argument('--num_layers', help='num of layers', default=1)
    parser.add_argument('--sentence_mode',
                        help='one sentence one event is Spe,one sentence may have many events is Full', default='Full')
    parser.add_argument('--labeled_data_path', help='labeled data path',
                        default=os.path.join(os.path.join(rootPath, 'labeled'), parser.get_default('sentence_mode')))
    parser.add_argument('--max_sequence_length', help='max length of sequence', default= 51 if parser.get_default('sentence_mode')=='Spe' else 40)  # Full - 40,Spe-51
    parser.add_argument('--batch_size', help='batch size', default=32)
    parser.add_argument('--num_epochs', help='num of epochs', default=2)
    parser.add_argument('--device_map', help='which device to see', default='CPU:0')
    parser.add_argument('--segmentor_model_path', help='segmentor model path',
                        default=os.path.join(ltpPath, 'cws.model'))
    parser.add_argument('--segmentor_user_dict_path', help='segmentor user dictionary path',
                        default=os.path.join(ltpPath, 'userDict.txt'))
    parser.add_argument('--word2vec_path', help='word2vecpath', default=os.path.join(rootPath, 'word2vec'))
    parser.add_argument('--embedded_dim', help='wordembeddeddim', default=300)
    FLAGS, args = parser.parse_known_args()
    return FLAGS ;

def train():
    FLAGS = getParser()
    FLAGS.ifTrain = True
    FLAGS.ifTest = True
    run.main(FLAGS)
def test():
    FLAGS = getParser()
    FLAGS.ifTrain = False
    FLAGS.ifTest = True
    run.main(FLAGS)

def predict(sentence):
    FLAGS = getParser()
    FLAGS.ifTrain = False
    FLAGS.ifTest = False
    FLAGS.ifPredict = True
    run.main(FLAGS,sentence)

#抽取事件、参数
def extractEvent(sentence):
    triggers = []
    FLAGS = getParser()
    # 获取触发词tag
    with open(os.path.join(FLAGS.root_dir,'triggerLabels.txt'), 'r', encoding='utf8') as f:
        for line in f.readlines():
            triggers.append(line.strip())
    FLAGS.ifTrain = False
    FLAGS.ifTest = False
    FLAGS.ifPredict = True
    words,tags = run.main(FLAGS, sentence)
    print(' '.join(words))
    print(' '.join(tags))
    events = []
    arguments = []
    for word,tag in zip(words,tags):
        if(tag in triggers):
            events.append(word)
        elif(tag !='O'):
            arguments.append(word)
    print(' '.join(events))
    print(' '.join(arguments))
if __name__=='__main__':
    train()
    # predict('被告季某辩称，原告所陈述的事实理由不正确，原被告于2009年农历正月认识，××××年××月××日生育一女，婚后为了抚养小孩发生了争吵，被告也曾不小心碰伤了原告。')

    # predict('原、被告于2007年11月于网上相识恋爱，200 8年3月17日登记结婚，××××年××月××日生育女儿戴某乙，2012 年6月1日生育女儿罗某乙。')
    sys.exit(0)