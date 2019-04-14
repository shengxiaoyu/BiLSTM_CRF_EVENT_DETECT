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






if __name__ == '__main__':
    train()
    # test()
    sys.exit(0)