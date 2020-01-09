#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import functools
import os

__doc__ = 'description:模型输入预处理中心'
__author__ = '13314409603@163.com'
from event_dectect.First_For_Commo_Tags import config_center as CONFIG
import tensorflow as tf
import numpy as np


# 对齐和向量化
def paddingAndEmbedding(fileName, words, tags, posTags, max_sequence_length, noEmbedding):
    # 如果出现非pos标签，则使用<pad>代替
    for index in range(len(posTags)):
        try:
            CONFIG.POS_2_ID[posTags[index]]
        except:
            posTags[index] = '<pad>'

    # 构造每个词是否是触发词的特征
    triggerFlags = [[1] if tag in CONFIG.TRIGGER_TAGs else [0] for tag in tags]

    # 转换为字为中心
    chars = []
    char_tags = []
    char_pos_tags = []
    char_trigger_tags = []
    chars_to_words = []
    for word, tag, posTag, triggerFlag in zip(words, tags, posTags, triggerFlags):
        for index, char in enumerate(word):
            chars.append(char)
            chars_to_words.append(word)
            if (tag.find('B_') != -1 and index != 0):
                char_tags.append(tag.replace('B_', 'I_'))
            else:
                char_tags.append(tag)

            char_pos_tags.append(posTag)

            char_trigger_tags.append(triggerFlag)

    # padding or cutting
    # print(fileName)
    length = len(chars)
    if (length < max_sequence_length):
        for i in range(length, max_sequence_length):
            # words.append('<pad>')
            chars.append('*')
            chars_to_words.append('<pad>')
            char_tags.append('<pad>')
            char_pos_tags.append('<pad>')
            char_trigger_tags.append([0])
    else:
        # words = words[:max_sequence_length]
        chars = chars[:max_sequence_length]
        chars_to_words = chars_to_words[:max_sequence_length]
        char_tags = char_tags[:max_sequence_length]
        char_pos_tags = char_pos_tags[:max_sequence_length]
        char_trigger_tags = char_trigger_tags[:max_sequence_length]

    # 处理意外词
    # 如果是词汇表中没有的词，则使用<pad>代替
    for index in range(min(length, max_sequence_length)):
        try:
            CONFIG.WV[chars_to_words[index]]
        except:
            chars_to_words[index] = '<pad>'
    for index in range(min(length, max_sequence_length)):
        try:
            CONFIG.CHAR_WV[chars[index]]
        except:
            chars[index] = '*'
    # postag 转id
    # 转one-hot
    char_pos_tags = [CONFIG.POS_2_ID[pos] if pos in CONFIG.POS_2_ID else 0 for pos in char_pos_tags]
    # 如果id不为0就转为正常的onehot，如果为0就转为全为0的onehot表示，这样去掉<pad>的onehot表示，只留正确意义的pos
    char_pos_tags = [[1 if i == id else 0 for i in CONFIG.POS_2_ID.values()] for id in char_pos_tags]

    # 根据noEmbedding参数确定是否进行向量化
    if (not noEmbedding):
        # words = [CONFIG.WV[word] for word in words]
        chars = [np.append(CONFIG.CHAR_WV[char], CONFIG.WV[word]) for char, word in zip(chars, chars_to_words)]

    try:
        char_tags = [CONFIG.TAG_2_ID[tag] for tag in char_tags]
    except:
        print('这个文件tag无法找到正确索引，请检查:' + fileName)

    return (chars, min(length, max_sequence_length), char_pos_tags, char_trigger_tags), char_tags


def generator_fn(input_dir, max_sequence_length, dirs, noEmbedding=False, sentences_words_posTags=None, ):
    result = []
    if (sentences_words_posTags):
        for one_sentence_words_posTags in sentences_words_posTags:
            result.append(paddingAndEmbedding('sentence', one_sentence_words_posTags[0], one_sentence_words_posTags[1],
                                              one_sentence_words_posTags[2], max_sequence_length, noEmbedding))
    elif (input_dir):
        dirs = dirs.split(',')
        for dir in os.listdir(input_dir):
            # 03 36 69
            dir_path = os.path.join(input_dir, dir)
            for sub_dir in os.listdir(dir_path):
                if (sub_dir not in dirs):
                    continue
                sub_dir_path = os.path.join(dir_path, sub_dir)
                for input_file in os.listdir(sub_dir_path):
                    with open(os.path.join(sub_dir_path, input_file), 'r', encoding='utf8') as f:
                        sentence = f.readline()  # 句子行
                        while sentence:
                            # 标记行
                            label = f.readline()
                            pos = f.readline()
                            if not label:
                                break
                            words = sentence.strip().split(' ')
                            words = list(filter(lambda word: word != '', words))

                            tags = label.strip().split(' ')
                            tags = list(filter(lambda word: word != '', tags))

                            posTags = pos.strip().split(' ')
                            posTags = list(filter(lambda word: word != '', posTags))

                            sentence = f.readline()

                            if (len(words) != len(tags) or len(tags) != len(posTags)):
                                print(input_file, ' words、labels、pos数不匹配：' + sentence + ' words length:' + str(
                                    len(words)) + ' labels length:' + str(len(tags)) + ' pos length:' + str(
                                    len(posTags)))
                                continue
                            result.append(
                                paddingAndEmbedding(input_file, words, tags, posTags, max_sequence_length, noEmbedding))
    return result


def input_fn(input_dir, shuffe, num_epochs, batch_size, max_sequence_length, dirs, sentences_words_posTags=None):
    shapes = (([max_sequence_length, CONFIG.CHAR_WV.vector_size + CONFIG.WV.vector_size], (),
               [max_sequence_length, CONFIG.POSs_LEN],
               [max_sequence_length, 1]), [max_sequence_length])
    types = ((tf.float32, tf.int32, tf.float32, tf.float32), tf.int32)
    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, input_dir=input_dir, sentences_words_posTags=sentences_words_posTags,
                          max_sequence_length=max_sequence_length, dirs=dirs),
        output_shapes=shapes,
        output_types=types
    )
    if shuffe:
        dataset = dataset.shuffle(buffer_size=25000).repeat(num_epochs)

    dataset = dataset.batch(batch_size)
    return dataset


def findTrigger(sentence):
    triggers = []
    for tag, words in CONFIG.TRIGGER_WORDS_DICT.items():
        for word in words:
            beginIndex = sentence.find(word)
            if (beginIndex != -1):
                endIndex = beginIndex + len(word)
                triggers.append([tag, beginIndex, endIndex])
    if (len(triggers) == 0):
        return None
    # 处理相互覆盖的触发词，如果找到的触发词有交集，就取最长的触发词
    # 现用beginIndex排序
    triggers.sort(key=lambda x: x[1])

    # 存储新的触发词
    newTriggers = []
    # 当前保留的最新的一个触发词
    currenctTrigger = triggers[0]
    for trigger in triggers:
        if (trigger[1] < currenctTrigger[2]):  # 如果下一个触发词的beginIndex小于最新触发词，则说明有交集
            if (trigger[2] - trigger[1] > currenctTrigger[2] - currenctTrigger[1]):
                currenctTrigger = trigger
        else:  # 无交集，则将当前触发词保存，说明该触发词没有覆盖其他触发词
            newTriggers.append(currenctTrigger)
            currenctTrigger = trigger
    newTriggers.append(currenctTrigger)
    return newTriggers


# 标注trigger触发词
def labelTrigger(words, labeled, beginIndex, endIndex, tag):
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
    return words, labeled


if __name__ == '__main__':
    str = "1,2,3,4,5,6"
    strs = str.split(',')
    s = set(strs)
    if ('1' in strs):
        print('success')
    if ('1' in s):
        print('success')
