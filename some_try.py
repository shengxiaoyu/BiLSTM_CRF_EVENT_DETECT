#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

__doc__ = 'description'
__author__ = '13314409603@163.com'

from pyltp import Postagger
from pyltp import Segmentor
# from Config import config_parser
from RestApi import service
# from Event_Model.extract_event import Extractor

LTP_DATA_DIR='C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\ltp_data_v3.4.0'
pos_mode_path = os.path.join(LTP_DATA_DIR,'pos.model')
segmentor_model_path = os.path.join(LTP_DATA_DIR,'cws.model')
user_dict_path = os.path.join(LTP_DATA_DIR,'userDict.txt')

def main():
    postagger = Postagger()
    postagger.load(pos_mode_path)

    segmentor = Segmentor()
    segmentor.load_with_lexicon(segmentor_model_path,user_dict_path)

    str = '原告韦某诉称，原告韦某与被告季某于2010年2月份经朋友介绍认识，双方认识一个月后即××××年××月××日便登记结婚。\n××××年××月××日双方生育一女，名为季韦唯。\n婚后双方经常为琐事争吵，被告经常对原告使用家庭暴力。'
    # str = str.replace('\n\r','**')
    # str = str.replace('\n','**')
    # words=['原告','董' ,'某某' ,'诉称' ,'原告', '被告' ,'经人介绍', '后', '2008年', '10月', '1日', '当地', '风俗', '举行', '结婚', '仪式', '后', '2009年', '2月', '10日', '补办',
    #        '结婚', '登记']
    words = segmentor.segment(str)
    words = list(words)
    result = postagger.postag(words)
    print(' '.join(result))
    postagger.release()

if __name__ == '__main__':
    # main()
    # print("end")
    service.app.run(host='127.0.0.1',port=8000)
    sys.exit(0)