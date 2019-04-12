#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__doc__ = 'description'
__author__ = '13314409603@163.com'


import os
import sys
from pyltp import Segmentor

import LSTM_CRF.word2vec_lstm_crf_ed as run
from Config.config_parser import getParser
from Word2Vec.my_word2vec import Word2VecModel
from data_pre_process.cpws_process import CpwsProcess
from data_pre_process.qsz_process import QszProcess
from data_pre_process.tsbl_process import TsblProcess


#从原材料中生成分句后的起诉庭审笔录、起诉状、裁判文书
def processData():
    FLAGS = getParser()
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
    FLAGS = getParser()
    #训练word2vec，传入path表示使用新训练集继续训练
    wvm = Word2VecModel(FLAGS.word2vec_path,os.path.join(FLAGS.word2vec_path,'train'),size=FLAGS.embedded_dim)
    if(new_data_path!=None):
        wvm.reTrain(new_data_path)
def testWore2vec():
    FLAGS = getParser()
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

#接收已分好句的句子进行预测抽取
def predict(sentences):
    FLAGS = getParser()
    FLAGS.ifTrain = False
    FLAGS.ifTest = False
    FLAGS.ifPredict = True
    return run.main(FLAGS,sentences)

#预测一个文件夹下所有文件内容，构造训练样本给下一层标注模型
def predictFile(dir):
    FLAGS = getParser()
    FLAGS.ifTrain = False
    FLAGS.ifTest = False
    FLAGS.ifPredict = False
    FLAGS.ifPredictFile = True
    run.main(FLAGS=FLAGS,dir=dir)

if __name__=='__main__':
    train()
    # test()
    # predict(['被告季某辩称，原告所陈述的事实理由不正确，原被告于2009年农历正月认识，××××年××月××日生育一女，婚后为了抚养小孩发生了争吵，被告也曾不小心碰伤了原告。'])
    # predict(['原、被告于2007年11月于网上相识恋爱，200 8年3月17日登记结婚，××××年××月××日生育女儿戴某乙，2012 年6月1日生育女儿罗某乙。'])
    # predictFile('C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\labeled\\Spe\\dev')
    # predictFile('C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\labeled\\Spe\\test')
    sys.exit(0)