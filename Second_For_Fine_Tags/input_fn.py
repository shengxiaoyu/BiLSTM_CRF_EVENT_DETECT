#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'

import os
import functools
import First_For_Commo_Tags.config_center as CONFIG
import Second_For_Fine_Tags.config_center as NEW_CONFIG
import tensorflow as tf

#对齐和向量化
def paddingAndEmbedding(fileName,words,tags,pre_tags,max_sequence_length,noEmbedding):
    # print(fileName)
    length = len(words)

    #处理意外词
    # 如果是词汇表中没有的词，则使用<pad>代替
    for index in range(length):
        try:
            CONFIG.WV[words[index]]
        except:
            words[index] = '<pad>'


    #padding or cutting
    if(length<max_sequence_length):
        for i in range(length,max_sequence_length):
            words.append('<pad>')
            tags.append('<pad>')
            pre_tags.append('<pad>')
    else:
        words = words[:max_sequence_length]
        tags = tags[:max_sequence_length]
        pre_tags = pre_tags[:max_sequence_length]



    #添加是否是所关注的触发词 特征
    triggerFeatures = []
    for tag in pre_tags:
        # new_trigger_onehot = [1 if tag==new_trigger else 0 for new_trigger in NEW_CONFIG.NEW_TRIGGER_TAGs ]
        triggerFeatures.append([1 if tag in NEW_CONFIG.NEW_TRIGGER_TAGs else 0])
        # triggerFeatures.append(new_trigger_onehot)
        # if(tag in NEW_CONFIG.NEW_TRIGGER_TAGs):
        #     type = tag[2:-8]
        #     triggerFeatures.append(NEW_CONFIG.NEW_TRIGGER_TYPE_ONEHOT_DICT[type])
        # else:
        #     triggerFeatures.append([0 for _ in range(len(NEW_CONFIG.NEW_TRIGGER_TYPE_ONEHOT_DICT))])

    #根据noEmbedding参数确定是否进行向量化
    if(not noEmbedding):
        words = [CONFIG.WV[word] for word in words]
        # 第一层的预测标签转onehot表示
        oneHotOldTags = []
        # for tag in tags:
        #     oneHotOldTags.append([1 if tag == oldTag else 0 for oldTag in CONFIG.ARGU_TAGs])
        allOldTags = []
        allOldTags.extend(CONFIG.ARGU_TAGs)
        allOldTags.extend(CONFIG.TRIGGER_TAGs)
        for tag in tags:
            oneHotOldTags.append([1 if tag == oldTag else 0 for oldTag in allOldTags])
    else:
        #第一层的预测标签不变
        oneHotOldTags = tags
    try:
        pre_tags = [NEW_CONFIG.NEW_TAG_2_ID[tag] for tag in pre_tags]
    except:
        print('这个文件tag无法找到正确索引，请检查:'+fileName)

    return (words,min(length,max_sequence_length),oneHotOldTags,triggerFeatures),pre_tags


def generator_fn(input_dir,max_sequence_length,noEmbedding=False,sentence_words_firstTags_trueTriggerTags=None):
    result = []
    if(sentence_words_firstTags_trueTriggerTags):
        for one_sentence_words_firstTags_trueTriggerTags in sentence_words_firstTags_trueTriggerTags:
            result.append(paddingAndEmbedding('sentence', one_sentence_words_firstTags_trueTriggerTags[0], one_sentence_words_firstTags_trueTriggerTags[1],one_sentence_words_firstTags_trueTriggerTags[2],max_sequence_length, noEmbedding))
    elif(input_dir):
        for input_file in os.listdir(input_dir):
            with open(os.path.join(input_dir,input_file),'r',encoding='utf8') as f:
                sentence = f.readline()#句子行
                while sentence:
                    #原词行
                    words = sentence.strip().split()
                    #第一个模型的预测标签行
                    tags = f.readline().strip().split()
                    #pos行
                    f.readline()
                    #针对单个触发词时的真实标签行
                    pre_tags = f.readline().strip().split()
                    sentence = f.readline()
                    if (len(words) != len(tags)):
                        print(input_file, ' words、labels数不匹配：' + sentence + ' words length:' + str(
                            len(words)) + ' labels length:' + str(len(tags)))
                        continue
                    result.append(paddingAndEmbedding(input_file,words,tags,pre_tags,max_sequence_length,noEmbedding))
    return result

def input_fn(input_dir,shuffe,num_epochs,batch_size,max_sequence_length,sentence_words_firstTags_trueTriggerTags=None):
    '''shape代表((最大句长，词向量长),真实句长，（最大句长，新Trigger类别),(最大句长，旧参数类别)),真实标签)'''
    shapes = (([max_sequence_length,CONFIG.WV.vector_size],(),[max_sequence_length,len(CONFIG.ARGU_TAGs)+len(CONFIG.TRIGGER_TAGs)],[max_sequence_length,1]),[max_sequence_length])
    types = ((tf.float32,tf.int32,tf.float32,tf.float32),tf.int32)
    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn,input_dir=input_dir,sentence_words_firstTags_trueTriggerTags=sentence_words_firstTags_trueTriggerTags,max_sequence_length = max_sequence_length),
        output_shapes=shapes,
        output_types=types
    )
    if shuffe:
        dataset = dataset.shuffle(buffer_size=20000).repeat(num_epochs)

    dataset = dataset.batch(batch_size)
    return dataset

if __name__ == '__main__':
    pass