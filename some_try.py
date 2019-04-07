#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

__doc__ = 'description'
__author__ = '13314409603@163.com'

from pyltp import Postagger

LTP_DATA_DIR='C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\ltp_data_v3.4.0'
pos_mode_path = os.path.join(LTP_DATA_DIR,'pos.model')

def main():
    postagger = Postagger()
    postagger.load(pos_mode_path)

    words=['原告','董' ,'某某' ,'诉称' ,'原告', '被告' ,'经人介绍', '后', '2008年', '10月', '1日', '当地', '风俗', '举行', '结婚', '仪式', '后', '2009年', '2月', '10日', '补办',
           '结婚', '登记']
    result = postagger.postag(words)
    print(' '.join(result))
    postagger.release()

if __name__ == '__main__':
    main()
    sys.exit(0)