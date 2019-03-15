#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import csv

__doc__ = 'description'
__author__ = '13314409603@163.com'

MAX_LENGTH = 1000
LENGTHS = [0 for _ in range(MAX_LENGTH)]
SUMS = [0 for _ in range(MAX_LENGTH)]
COUNT = 1
#统计句子长度分布
def calSentenceLength(path):
    if(os.path.isdir(path)):
        for subPath in os.listdir(path):
            calSentenceLength(os.path.join(path,subPath))
    else:#统计单个文件
        with open(path, 'r', encoding='utf8') as f:
            for sentence in f.readlines():
                length = len(sentence.strip().split(' '))
                LENGTHS[length] += 1

#输入占比区间，例如输入0.95，返回包含95%的句子此时需要设置句子长度的阈值
def getMax(rate):
    sum = 0
    for index,count in enumerate(LENGTHS):
        sum += count
        SUMS[index] = sum
    for index in range(MAX_LENGTH):
        if(SUMS[index]/SUMS[-1]>=rate):
            return index

#将案号转为indxe 文件名称
def an2Index(path):
    if(not os.path.exists(path)):
        return
    elif(os.path.isdir(path)):
        for file in os.listdir(path):
            an2Index(os.path.join(path,file))
    else:
        global COUNT
        os.renames(path,os.path.join(os.path.dirname(path),str(COUNT)+'.txt'))
        COUNT += 1




if __name__ == '__main__':
    an2Index('C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\segment_result\\qstsbl')
    sys.exit(0)