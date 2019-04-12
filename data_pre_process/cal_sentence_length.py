#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pyltp import Segmentor
__doc__ = 'description'
__author__ = '13314409603@163.com'

base_dir =  'C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF'
ltpPath = os.path.join(base_dir, 'ltp_data_v3.4.0')

MAX_LENGTH = 1000
LENGTHS = [0 for _ in range(MAX_LENGTH)]
SUMS = [0 for _ in range(MAX_LENGTH)]
COUNT = 1

# segmentor = Segmentor()
# segmentor.load_with_lexicon(os.path.join(ltpPath,'cws.model'), os.path.join(ltpPath,'userDict.txt'))
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

TRIGGERs = []
ARGUs = []
EVENTs = {}
def initTags():
    global ARGUs,TRIGGERs
    with open(os.path.join(base_dir,'argumentLabels.txt'), 'r', encoding='utf8') as f:
        for line in f.readlines():
            ARGUs.append(line.strip())
    #获取触发词tag
    with open(os.path.join(base_dir,'triggerLabels.txt'), 'r', encoding='utf8') as f:
        for line in f.readlines():
            TRIGGERs.append(line.strip())
#统计各类事件参数位置分布
def calEventArguDis(path):
    def handlerSingleFile(path):
        with open(path,'r',encoding='utf8') as f:
            sentence = f.readline()
            while(sentence):
                label = f.readline()
                pos = f.readline()
                tags = label.split()
                #触发词
                trigger = None
                triggerIndex = None
                for index,tag in enumerate(tags):
                    if(tag in TRIGGERs ):
                        trigger = tag[2:]#去掉B_,I_
                        triggerIndex = index
                        if(trigger not in EVENTs):
                            EVENTs[trigger] = {}
                if(not trigger):
                    sentence = f.readline()
                    continue
                #遍历参数
                for index,tag in enumerate(tags):
                    if(tag in ARGUs and tag!='O'):
                        argu = tag[2:] #去掉B_,I_
                        if(argu not in EVENTs[trigger]):
                            EVENTs[trigger][argu] = {}
                        pos = index-triggerIndex
                        if(pos not in EVENTs[trigger][argu]):
                            EVENTs[trigger][argu][pos] = 1
                        else:
                            EVENTs[trigger][argu][pos] += 1
                sentence = f.readline()

    if(os.path.isdir(path)):
        for file in os.listdir(path):
            calEventArguDis(os.path.join(path,file))
    else:
        handlerSingleFile(path)

#统计各类tag可能的pos标注类型
def calRelationOfTagAndPos(path):
    tags_poses = {}
    def handlerSingleFile(path):
        with open(path,'r',encoding='utf8') as f:
            sentence = f.readline()
            while(sentence):
                tags = f.readline().strip().split()
                poses = f.readline().strip().split()
                for tag,pos in zip(tags,poses):
                    if(tag!='O'):
                        if(tag not in tags_poses):
                            tags_poses[tag] = {}
                        if(pos not in tags_poses[tag]):
                            tags_poses[tag][pos] = 0
                        tags_poses[tag][pos] += 1
                sentence = f.readline()
    def handlerDir(path):
        for subPath in os.listdir(path):
            newPath = os.path.join(path,subPath)
            if(os.path.isdir(newPath)):
                handlerDir(newPath)
            else:
                handlerSingleFile(newPath)
    if(os.path.isdir(path)):
        handlerDir(path)
    else:
        handlerSingleFile(path)
    print(tags_poses)

    dir = path

    for tag,poses in tags_poses.items():
        with open(os.path.join(dir,tag+'.txt'),'w',encoding='utf8') as fw:
            ss = list(poses.items())
            ss.sort(key=lambda x:x[1])
            ss.reverse()
            for pos,count in ss:
                fw.write(pos+'\t'+str(count)+'\n')

def getMaxSentenceLength(path):
    maxLength = 0
    for file in os.listdir(path):
        with open(os.path.join(path,file),'r',encoding='utf8') as f:
            line = f.readline()
            while(line):
                words = f.readline().strip().split()
                if(len(words)>40):
                    print(len(words))
                maxLength = max(maxLength,len(f.readline().strip().split()))

    print(maxLength)
if __name__ == '__main__':
    # calSentenceLength('C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\labeled\\Full')
    # print(getMax(0.95))
    # calSentenceLength('C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\labeled\\Spe')
    # print(getMax(0.95))

    # initTags()
    # calEventArguDis(os.path.join(os.path.join(base_dir,'labeled'),'Spe'))
    # print(EVENTs)
    # trigger_argu_dis_path = os.path.join(base_dir,'trigger_argus')
    # if (not os.path.exists(trigger_argu_dis_path)):
    #     os.mkdir(trigger_argu_dis_path)
    # for (trigger,arugs) in EVENTs.items():
    #     for(argu,pos_count) in arugs.items():
    #         with open(os.path.join(trigger_argu_dis_path,trigger+'_'+argu+'.txt'),'w',encoding='utf8') as f:
    #             for (pos,count) in pos_count.items():
    #                 f.write(str(pos)+'\t'+str(count)+'\n')

    calRelationOfTagAndPos('C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\labeled\\Full')

    sys.exit(0)