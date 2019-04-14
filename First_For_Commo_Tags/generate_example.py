#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

import First_For_Commo_Tags.config_center as CONFIG
from Config.config_parser import getParser
__doc__ = 'description,将Full'
__author__ = '13314409603@163.com'


#将Spe模式中同一个句子的触发词合到一个句子中，构造第一个模型的输入
def formExamples(length,tags_list):
    tags_for_first_model = ['O' for _ in range(length)]
    for old_tags in tags_list:
        for index,tag in enumerate(old_tags):
            if(tag.find('Trigger')!=-1):
                tag_for_first_model = tag[0:-8] #去掉_Trigger就变成了原先的tag
                tags_for_first_model[index]=tag_for_first_model
    return tags_for_first_model

#构造第一个模型的输入样例和第二个模型的标注结果
def generator_examples_from_full_file(path):
    sentence_words_oldTags_poses = [] #存储构造的第一个模型输入
    sentence_fullTags = [] #存储第二个模型的真实输出和输入给第一个模型预测样例的id映射
    index = 0
    for fileName in os.listdir(path):
        with open(os.path.join(path,fileName),'r',encoding='utf8') as f:
            #初始化
            lastSentence = f.readline()
            lastWords = lastSentence.strip().split()
            tags_list = []
            tags_list.append(f.readline().strip().split())
            lastPoses = f.readline().strip().split()
            length = len(lastWords)

            #新句子
            sentence = f.readline()
            while(sentence):
                if(sentence==lastSentence):
                    #是同一个句子，把tags加入队列
                    tags_list.append(f.readline().strip().split())
                    #把重复的poses去掉
                    f.readline()
                    #更新句子
                    sentence = f.readline()
                else:
                    #出现了新的句子，则先把原句子处理了
                    first_tags = formExamples(length,tags_list)
                    sentence_words_oldTags_poses.append([lastWords,first_tags,lastPoses])
                    for second_tags in tags_list:
                        sentence_fullTags.append([second_tags,index])
                    index += 1

                    #重新初始化
                    lastSentence = sentence
                    lastWords = lastSentence.strip().split()
                    tags_list = []
                    tags_list.append(f.readline().strip().split())
                    lastPoses = f.readline().strip().split()
                    length = len(lastWords)

                    sentence=f.readline()

            #遍历完把最后缓存中的处理了
            first_tags = formExamples(length, tags_list)
            sentence_words_oldTags_poses.append([lastWords, first_tags, lastPoses])
            for second_tags in tags_list:
                sentence_fullTags.append([second_tags, index])
            index += 1
    return sentence_words_oldTags_poses,sentence_fullTags
#生成精细的标签集合
def generator_full_tags():
    FLAGS = getParser()
    triggerLabelsPath = os.path.join(FLAGS.root_dir,'triggerLabels.txt')
    arguPath = os.path.join(FLAGS.root_dir,'full_tags.txt')
    with open(triggerLabelsPath,'r',encoding='utf8') as f,open(os.path.join(os.path.split(triggerLabelsPath)[0],'full_trigger_labels.txt'),'w',encoding='utf8') as fw:
        fw.write('\n'.join(list(map(lambda line:line.strip()+'_Trigger',f.readlines()))))
    with open(arguPath,'r',encoding='utf8') as f,open(os.path.join(os.path.split(arguPath)[0],'full_argu_labels.txt'),'w',encoding='utf8') as fw:
        argus = [line.strip() for line in f.readlines()]
        for argu in argus:
            fw.write('B_' + argu + '\n' + 'I_' + argu + '\n')
        fw.write('O')
#将spe模型下的单句合并为full下的句子
def merge(path):
    #会用到trigger集合，需要初始化Trigger_Tags
    parse = getParser()
    CONFIG.init(parse.root_dir)
    #新文件的保存路径，保存在传入文件的同级目录
    savePath = os.path.join(os.path.split(path)[0],'Merge_'+os.path.split(path)[1])
    if(not os.path.exists(savePath)):
        os.mkdir(savePath)

    def merge(tagsList):
        mergedTags = tagsList[0]
        for tags in tagsList[1:]:
            for index,tag in enumerate(tags):
                if(tag!='O'):
                    if(mergedTags[index]=='O'):
                        mergedTags[index] = tag
                    elif(mergedTags[index] in CONFIG.TRIGGER_TAGs):
                        '''此时产生冲突'''
                        '''原先填入的是触发词'''
                        if(mergedTags[index].find('B_')==-1 and tag.find('B_')!=-1):
                            mergedTags[index] = tag #原先的不是B_开头触发词，新来的是B_开头触发词才能覆盖
                    else:#如果以前不是触发词，
                        if((tag in CONFIG.TRIGGER_TAGs or tag.find('B_')!=-1) and mergedTags[index].find('B_')==-1): #只有新来的是触发词或者B_开头的参数，而且老的不是B_开头才能覆盖
                            mergedTags[index] = tag
        return mergedTags

    for fileName in os.listdir(path):
        with open(os.path.join(path,fileName),'r',encoding='utf8') as f,open(os.path.join(savePath,fileName),'w',encoding='utf8') as fw:
            #words行
            lastSentence = f.readline().strip()
            #tag行
            lastTagsList = []
            lastTags = f.readline().strip().split()
            lastTagsList.append(lastTags)
            #pos行
            poses = f.readline().strip()

            sentence = f.readline().strip()
            while(sentence):
                if(sentence==lastSentence):
                    '''此时时同一行，将tags加入列表'''
                    lastTagsList.append(f.readline().strip().split())
                    #去掉pos行
                    f.readline()
                    #更新words行
                    sentence = f.readline().strip()
                else:
                    '''来了新的行，将上一种合并写入'''
                    mergedTags = merge(lastTagsList)
                    fw.write(lastSentence+'\n'+' '.join(mergedTags)+'\n'+poses+'\n')
                    #更新缓存
                    lastSentence = sentence
                    lastTagsList = []
                    lastTags = f.readline().strip().split()
                    lastTagsList.append(lastTags)
                    poses = f.readline().strip()

                    sentence = f.readline().strip()

            #处理缓存
            mergedTags = merge(lastTagsList)
            fw.write(lastSentence+'\n'+' '.join(mergedTags)+'\n'+poses+'\n')

if __name__ == '__main__':
    merge('C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\labeled\\Spe\\train')
    # generator_full_tags()
    sys.exit(0)