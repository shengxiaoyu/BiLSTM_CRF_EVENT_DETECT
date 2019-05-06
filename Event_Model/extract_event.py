#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

__doc__ = 'description'
__author__ = '13314409603@163.com'
from pyltp import SentenceSplitter

import First_For_Commo_Tags.config_center as CONFIG
import First_For_Commo_Tags.word2vec_lstm_crf_ed as first
import Second_For_Fine_Tags.word2vec_lstm_crf_argu_match as second
from Config.config_parser import getParser
import Event_Model.EventModel as EventModel
import Second_For_Fine_Tags.config_center as NEW_CONFIG

root_dir = 'C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF'

#判断是否含有关注事实触发词



class Extractor(object):
    def __init__(self):
        print('构造参数')
        self.FLAGS = getParser()
        self.FLAGS.ifTrain = False
        self.FLAGS.ifTest = False
        self.FLAGS.ifPredict = True
        CONFIG.init(self.FLAGS.root_dir)
        NEW_CONFIG.initNewTags(os.path.join(self.FLAGS.root_dir, 'full_trigger_labels.txt'),
                               os.path.join(self.FLAGS.root_dir, 'full_argu_labels.txt'))
        self.sentenceSplitter = SentenceSplitter()

    def ifContainTrigger(self,sentence):
        # 初始化触发词集
        triggerDict = CONFIG.TRIGGER_WORDS_DICT
        # 判断释放含触发词
        triggerContained = ''
        for oneKindTrigger in triggerDict.items():
            triggerType = oneKindTrigger[0]
            for word in oneKindTrigger[1]:
                if (sentence.find(word) != -1):
                    triggerContained = triggerContained + (triggerType + ':' + word)
                    break
        if (len(triggerContained) == 0):
            return False
        return True
    def extractor(self,paragraph):
        # 调用预测接口

        sentences = []
        for sentence in self.sentenceSplitter.split(paragraph):
            hasTrigger = self.ifContainTrigger(sentence)
            if (not hasTrigger):
                print('该句子中无关注事实：' + sentence)
            else:
                sentences.append(sentence)
        print('第一个模型预测')
        if(len(sentences)==0):
            print("整个抽取文本无关注事实")
            return []
        words_list,first_tags_list,index_pairs_list = first.main(self.FLAGS, sentences)
        print('第二个模型预测')
        words_list,second_tags_list,index_pairs_list,sentences = second.main(self.FLAGS,words_firstTags_indxPairs_sentences=[words_list,first_tags_list,index_pairs_list,sentences])

        events = []
        for tags, words,index_pairs,sentence in zip(second_tags_list, words_list,index_pairs_list,sentences):
            event_argus_dict = {}
            event_argus_index_pair_dict = {}
            #针对每一句构造一个map
            for tag,word,index_pair in zip(tags,words,index_pairs):
                if (tag in NEW_CONFIG.NEW_TRIGGER_TAGs):
                    '''触发词标签，确定触发词和事件类型'''
                    if ('Trigger' in event_argus_dict):
                        '''将I_开头的加入'''
                        event_argus_dict['Trigger'] = event_argus_dict['Trigger'] + word
                        #索引的融合，默认是连在一起的，直接替换endIndex
                        last_index_pair = event_argus_index_pair_dict['Trigger']
                        event_argus_index_pair_dict['Trigger'] = [last_index_pair[0],index_pair[1]]
                    else:
                        event_argus_dict['Type'] = tag[2:-8] # B_Know_Trigger => Know
                        event_argus_dict['Trigger'] = word
                        event_argus_index_pair_dict['Trigger'] = index_pair
                if (tag in NEW_CONFIG.NEW_ARGU_TAGs and tag != '<pad>' and tag != 'O'):
                    newTag = tag[2:] #B_Argu => Argu
                    if (newTag in event_argus_dict):
                        event_argus_dict[newTag] = event_argus_dict[newTag] + word
                        last_index_pair = event_argus_index_pair_dict[newTag]
                        event_argus_index_pair_dict[newTag] = [last_index_pair[0],index_pair[1]]
                    else:
                        event_argus_dict[newTag] = word
                        event_argus_index_pair_dict[newTag] = index_pair


            if ('Type' in event_argus_dict):
                events.append(EventModel.EventFactory(event_argus_dict,event_argus_index_pair_dict,sentence))
        print('事件抽取完成：\n'+'\n'.join([str(event) for event in events]))
        return events
if __name__ == '__main__':
    # events = Extractor.extractor2('原、被告双方1986年上半年经人介绍认识，××××年××月××日在临桂县宛田乡政府登记结婚，××××年××月××日生育女儿李某乙，××××年××月××日生育儿子李某丙，现女儿李某乙、儿子李某丙都已独立生活')
    print('end')
    exit(0)