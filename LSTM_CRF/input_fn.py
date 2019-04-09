#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import functools
import os

__doc__ = 'description:模型输入预处理中心'
__author__ = '13314409603@163.com'
import LSTM_CRF.config_center as CONFIG
import tensorflow as tf

#对齐和向量化
def paddingAndEmbedding(fileName,words,tags,posTags,max_sequence_length,noEmbedding):
    # print(fileName)
    length = len(words)

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

    #padding or cutting
    if(length<max_sequence_length):
        for i in range(length,max_sequence_length):
            words.append('<pad>')
            tags.append('<pad>')
            posTags.append('<pad>')
    else:
        words = words[:max_sequence_length]
        tags = tags[:max_sequence_length]
        posTags = posTags[:max_sequence_length]

    #postag 转id
    posTags = [CONFIG.POS_2_ID[pos] for pos in posTags]
    #转one-hot
    posTags = [[1 if i==id else 0 for i in CONFIG.POS_2_ID.values()] for id in posTags]

    #根据noEmbedding参数确定是否进行向量化
    if(not noEmbedding):
        words = [CONFIG.WV[word] for word in words]
    try:
        tags = [CONFIG.TAG_2_ID[tag] for tag in tags]
    except:
        print('这个文件tag无法找到正确索引，请检查:'+fileName)

    return (words,min(length,max_sequence_length),posTags),tags


def generator_fn(input_dir,max_sequence_length,noEmbedding=False,sentences_words_posTags=None,):
    result = []
    if(sentences_words_posTags):
        for one_sentence_words_posTags in sentences_words_posTags:
            tags = ['O' for _ in range(len(one_sentence_words_posTags[0]))]
            result.append(paddingAndEmbedding('sentence', one_sentence_words_posTags[0], tags, one_sentence_words_posTags[1], max_sequence_length, noEmbedding))
    elif(input_dir):
        for input_file in os.listdir(input_dir):
            with open(os.path.join(input_dir,input_file),'r',encoding='utf8') as f:
                sentence = f.readline()#句子行
                while sentence:
                    #标记行
                    label = f.readline()
                    pos = f.readline()
                    if not label:
                        break
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
    shapes = (([max_sequence_length,CONFIG.WV.vector_size],(),[max_sequence_length,len(CONFIG.POS_2_ID)]),[max_sequence_length])
    types = ((tf.float32,tf.int32,tf.float32),tf.int32)
    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn,input_dir=input_dir,sentences_words_posTags=sentences_words_posTags,max_sequence_length = max_sequence_length),
        output_shapes=shapes,
        output_types=types
    )
    if shuffe:
        dataset = dataset.shuffle(buffer_size=10000).repeat(num_epochs)

    dataset = dataset.batch(batch_size)
    return dataset


if __name__ == '__main__':
    pass