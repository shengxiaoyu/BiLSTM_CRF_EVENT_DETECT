#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from Config.config_parser import getParser
from Second_For_Fine_Tags import word2vec_lstm_crf_argu_match as run
import sys
__doc__ = 'description'
__author__ = '13314409603@163.com'

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
def predict(sentencs_words_firstTags_list):
    FLAGS = getParser()
    FLAGS.ifTrain = False
    FLAGS.ifTest = False
    FLAGS.ifPredict = True
    run.main(FLAGS,sentencs_words_firstTags_list=sentencs_words_firstTags_list)


if __name__ == '__main__':
    train()
    # test()
    # words = '原告 王 某某 诉称 被告 2010年 4月 人 介绍 相识 同年 12月 17日 办理 结婚 登记 2011年 1月 28日 生育 子 取名 罗某甲'.split()
    # firstTags = 'O O O O O B_Time I_Time O O B_Know B_Time I_Time I_Time B_Marry I_Marry I_Marry B_Time I_Time I_Time B_Bear B_Gender O B_Name'.split()
    # predict([[words],[firstTags]])
    sys.exit(0)