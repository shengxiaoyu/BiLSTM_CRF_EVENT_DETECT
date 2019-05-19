#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'

import sys
from enum import Enum
import re
from First_For_Commo_Tags import config_center as CONFIG

class regularPattern(Enum):
    TIME = r'[0-9|X|×]{1,4}年[阴|农|古]?[历]?[0-9|X|×]{1,2}月[0-9|X|×]{1,2}日|[0-9|X|×]{1,4}年[阴|农|古]?[历]?[0-9|X|×]{1,2}月[份]?[初]?|[0-9|X|×]{4}年[夏初]?|' \
           r'[同|明|次|今|去|前|后]年[农|古]?[历]?[0-9]{1,2}月[份]?[初]?[0-9]{1,2}日|' \
           r'[同|明|次|今|去|前|后]年[农|古]?[历]?[0-9]{1,2}月[份]?[初]?|' \
           r'[同|明|次|今|去|前|后]年[初]?|' \
           r'[0-9]{1,2}月[份]?[初]?[0-9]{1,2}日|[0-9]{1,2}月[份]?[初]?|[0-9]{1,2}日'
    MONEY = r'[0-9]+[多]?[万]?元'
    NAME = r'[\u4e00-\u9fa5]{1}[某]{1,2}[甲|乙|丙|丁]?'
    # DOCUMENT = r'[\(|（][0-9]{4}[\)|）][\u4e00-\u9fa5]{1,2}[0-9]*民初[子]?[第]?[0-9]*号民事判决书|[\(|（][0-9]{4}[\)|）][\u4e00-\u9fa5]{1,2}[0-9]*民初[子]?[第]?[0-9]*号民事裁定书|[\(|（][0-9]{4}[\)|）][\u4e00-\u9fa5]{1,2}[0-9]*民初[子]?[第]?[0-9]*号民事调解书'
    DOCUMENT = r'[\(|（][0-9]{4}[\)|）][\u4e00-\u9fa5]{1,2}[0-9]*民初[子]?[字]?[第]?[0-9]*号[《]?[民事判决调解裁定书]{3,5}[》]?'

class mySegmentor(object):
    def __init__(self,segmentor):
        self.segmentor = segmentor
        self.patterns = [re.compile(pattern.value) for pattern in regularPattern]

    def segment(self,sentence):
        # 去掉空格
        sentence = sentence.replace(' ', '')
        words = list(self.segmentor.segment(sentence))
        for pattern in self.patterns:

            #正则匹配中该保留的分词
            key_words = pattern.findall(sentence)
            for key_word in key_words:
                #记录该保留词在此时分词后词语的起止索引
                begin_index_in_words = 0
                end_index_in_words = 0

                #记录该保留词在原句中的起始字符索引
                begin_index_in_sentence = sentence.find(key_word)
                end_index_in_sentence = begin_index_in_sentence + len(key_word)

                if(begin_index_in_sentence!=-1):
                    #遍历此时分词结果中的词语时,在原句中的字符索引
                    cusor = 0

                    #获得该保留词在分词后词组中的起始索引
                    while(cusor<begin_index_in_sentence):
                        cusor += len(words[begin_index_in_words])
                        begin_index_in_words+=1

                    #更新分词数组中的截止词索引
                    end_index_in_words = begin_index_in_words

                    if(cusor>begin_index_in_sentence):
                        begin_index_in_words-=1
                    # 获得分词词组中的截止词索引
                    cusor += len(words[end_index_in_words])
                    while(cusor<end_index_in_sentence):
                        end_index_in_words+=1
                        cusor+=len(words[end_index_in_words])
                    # 新的分词结果
                    new_words = []
                    new_key_word = ''
                    for index,word in enumerate(words):
                        if(index<begin_index_in_words or index>end_index_in_words):
                            new_words.append(word)
                        else:
                            if(index>=begin_index_in_words and index<=end_index_in_words):
                                new_key_word = new_key_word + word
                            if(index==end_index_in_words):
                                new_words.append(new_key_word)
                    words = new_words
        return words

    def release(self):
        self.segmentor.release()



def main():
    sentence = '原告诉称，原、被告经人介绍相识两个月后于××××年××月××日在高阳县民政局办理结婚登记，'
    sentence2 = '被告裴某辩称：同意离婚，贵院做出(2017)津0115民初8476号民事判决书，判决驳回原告离婚之诉讼请求.（2013）南法民初字第03183号《民事判决书》判决驳回封某某的诉讼请求'
    # CONFIG.init(r'C:\Users\13314\Desktop\Bi-LSTM+CRF')
    # segmentor = CONFIG.SEGMENTOR
    # words = segmentor.segment(sentence2)
    # pattern = re.compile(regularPattern.DOCUMENT.value)
    # words = pattern.findall(sentence2)
    # print('\n'.join(words))
    sentence = 'In the process of hearing the case, the judeg must have a clear understanding of it. If the core events, which judges should pay attention to when they are hearing a certain kind of cases, can be automatically extracted from the materials in the cases to construct unified cases’ models, not only can judges greatly improve efficiency, but also the possibility of misjudgment can be reduced. Howerver, in existing event extraction tasks, there is no complete and accurate definition of the core events types for the judicial field. Besides, in the materials in the cases, a single sentence may contain multiple events which share same parameters or trigger words. In response to the above problems, this paper presents a framework which includes the definition of the core event types of cases which belongs to a specified cause of dispute and a re-tagging method to accurately extract events from materials in the cases. Taking civil divorce cases as example, we use this framework to define the target event types. And then we realize a system that can automatically extract these target events. Experimental results demonstrate that this framework can accurately obtain the core events, and extract multiple events sharing parameters or sharing trigger words in a single sentence.'
    print(len(sentence))


if __name__ == '__main__':
    main()
    sys.exit(0)
