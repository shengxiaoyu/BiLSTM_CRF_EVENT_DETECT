#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__doc__ = 'description:模型输入预处理中心'
__author__ = '13314409603@163.com'
import functools
import os
import First_For_Commo_Tags.config_center as CONFIG
import tensorflow as tf
import copy
import re
#对齐和向量化
def paddingAndEmbedding(fileName,words,tags,posTags,max_sequence_length,noEmbedding):
    # print(fileName)
    length = len(words)

    #处理意外词
    # 如果是词汇表中没有的词，则使用<pad>代替
    for index in range(length):
        try:
            CONFIG.WV[words[index]]
        except:
            words[index] = '<pad>'
    # 如果出现非pos标签，则使用<pad>代替
    for index in range(length):
        try:
            CONFIG.POS_2_ID[posTags[index]]
        except:
            posTags[index] = '<pad>'

    #构造每个词是否是触发词的特征
    triggerFlags = [[1] if tag in CONFIG.TRIGGER_TAGs else [0] for tag in tags]


    #padding or cutting
    if(length<max_sequence_length):
        for i in range(length,max_sequence_length):
            words.append('<pad>')
            tags.append('<pad>')
            posTags.append('<pad>')
            triggerFlags.append([0])
    else:
        words = words[:max_sequence_length]
        tags = tags[:max_sequence_length]
        posTags = posTags[:max_sequence_length]
        triggerFlags = triggerFlags[:max_sequence_length]

    #postag 转id
    #转one-hot
    posTags = [CONFIG.POS_2_ID[pos] if pos in CONFIG.POS_2_ID else 0 for pos in posTags]
    #如果id不为0就转为正常的onehot，如果为0就转为全为0的onehot表示，这样去掉<pad>的onehot表示，只留正确意义的pos
    posTags = [[1 if i==id else 0 for i in CONFIG.POS_2_ID.values()] for id in posTags]

    #根据noEmbedding参数确定是否进行向量化
    if(not noEmbedding):
        words = [CONFIG.WV[word] for word in words]
    try:
        tags = [CONFIG.TAG_2_ID[tag] for tag in tags]
    except:
        print('这个文件tag无法找到正确索引，请检查:'+fileName)

    return (words,min(length,max_sequence_length),posTags,triggerFlags),tags


def generator_fn(input_dir,max_sequence_length,noEmbedding=False,sentences_words_posTags=None):
    result = []
    if(sentences_words_posTags):
        for one_sentence_words_posTags in sentences_words_posTags:
            result.append(paddingAndEmbedding('sentence', one_sentence_words_posTags[0], one_sentence_words_posTags[1], one_sentence_words_posTags[2], max_sequence_length, noEmbedding))
    elif(input_dir):
        for input_file in os.listdir(input_dir):
            with open(os.path.join(input_dir,input_file),'r',encoding='utf8') as f:
                sentence = f.readline()#句子行
                while sentence:
                    # 细粒度标记行
                    label = f.readline()
                    if not label:
                        break
                    pos = f.readline()

                    words = sentence.strip().split(' ')
                    words = list(filter(lambda word:word!='',words))

                    tags = label.strip().split(' ')
                    tags = list(filter(lambda word:word!='',tags))

                    posTags = pos.strip().split(' ')
                    posTags = list(filter(lambda word:word!='',posTags))

                    sentence = f.readline()

                    if (len(words) != len(tags) or len(tags)!=len(posTags)):
                        print(input_file, ' words、labels、pos数不匹配：' + sentence + ' words length:' + str(
                            len(words)) + ' labels length:' + str(len(tags))+' pos length:'+str(len(posTags)))
                        continue
                    result.append(paddingAndEmbedding(input_file,words,tags,posTags,max_sequence_length,noEmbedding))
    return result

def input_fn(input_dir,shuffe,num_epochs,batch_size,max_sequence_length,sentences_words_posTags=None):
    shapes = (([max_sequence_length,CONFIG.WV.vector_size],(),[max_sequence_length,CONFIG.POSs_LEN],[max_sequence_length,1]),[max_sequence_length])
    types = ((tf.float32,tf.int32,tf.float32,tf.float32),tf.int32)
    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn,input_dir=input_dir,sentences_words_posTags=sentences_words_posTags,max_sequence_length = max_sequence_length),
        output_shapes=shapes,
        output_types=types
    )
    if shuffe:
        dataset = dataset.shuffle(buffer_size=20000).repeat(num_epochs)

    dataset = dataset.batch(batch_size)
    return dataset

#查找所有的触发词
def findTrigger(sentence):
    triggers = []
    for tag,words in CONFIG.TRIGGER_WORDS_DICT.items():
        for word in words:
            start = 0
            beginIndex = sentence.find(word,start)
            while(beginIndex!=-1):
                endIndex = beginIndex+len(word)
                triggers.append([tag,beginIndex,endIndex])
                start = endIndex
                beginIndex = sentence.find(word,start)

    if(len(triggers)==0):
        return None
    #处理相互覆盖的触发词，如果找到的触发词有交集，就取最长的触发词
    #现用beginIndex排序
    triggers.sort(key=lambda x:x[1])

    #存储新的触发词
    newTriggers  = []
    #当前保留的最新的一个触发词
    currenctTrigger = triggers[0]
    for trigger in triggers:
        if(trigger[1]<currenctTrigger[2]):#如果下一个触发词的beginIndex小于最新触发词，则说明有交集
            if(trigger[2]-trigger[1]>currenctTrigger[2]-currenctTrigger[1]):
                currenctTrigger = trigger
        else:#无交集，则将当前触发词保存，说明该触发词没有覆盖其他触发词
            newTriggers.append(currenctTrigger)
            currenctTrigger = trigger
    newTriggers.append(currenctTrigger)
    return newTriggers
# 标注trigger触发词
def labelTrigger(words,labeled,beginIndex,endIndex,tag):
    words = copy.copy(words)
    labeled = copy.copy(labeled)
    coursor = 0
    isBegin = True
    for index, word in enumerate(words):
        beginCoursor = coursor
        endCoursor = len(word) + coursor
        if ((beginCoursor <= beginIndex and beginIndex < endCoursor) or
                (beginCoursor < endIndex and endIndex <= endCoursor) or
                (beginCoursor >= beginIndex and endCoursor <= endIndex)):
            # 此时进入范围
            if (labeled[index].find('O') != -1):
                if (isBegin):
                    label = 'B_' + tag
                    labeled[index] = label
                    isBegin = False
                else:
                    label = 'I_' + tag
                    labeled[index] = label
        coursor = endCoursor
    return words,labeled

if __name__ == '__main__':
    pass