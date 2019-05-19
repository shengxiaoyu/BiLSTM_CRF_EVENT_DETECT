#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'
from pyltp import Segmentor,Postagger
import os
from Config.regular import mySegmentor

ifInited = False

root_dir = r'C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\'

pyltp_dir = os.path.join(root_dir,'ltp_data_v3.4.0')

#分词器
    #载入模型
print('初始化分词器')
my_segmentor = Segmentor()
    #加载自定义词典
my_segmentor.load_with_lexicon(os.path.join(pyltp_dir,'cws.model'), os.path.join(pyltp_dir,'userDict.txt'))
    #初始化基于正则优化后的分词器
# my_segmentor = mySegmentor(my_segmentor)

#POS 标注模型
my_postagger = Postagger()
my_postagger.load(os.path.join(pyltp_dir, 'pos.model'))





if __name__ == '__main__':
    pass