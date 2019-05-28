#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import functools
import sys

from Config.config_parser import getParser

__doc__ = 'description'
__author__ = '13314409603@163.com'
from pyltp import SentenceSplitter

import First_For_Commo_Tags.config_center as CONFIG
import tensorflow as tf
import os
import First_For_Commo_Tags.model_fn as MODEL
import First_For_Commo_Tags.input_fn as INPUT
from Extract_Event.EventModel import EventFactory2

class Event_Detection(object):
    def __init__(self,FLAGS,output_path=None):
        self.FLAGS = FLAGS
        self.output_path = os.path.join(self.FLAGS.root_dir,output_path)
        self.__initFirstModel__()

    def __initFirstModel__(self):
        FLAGS = self.FLAGS
        tf.enable_eager_execution()
        # 配置哪块gpu可见
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device_map

        # 在re train 的时候，才删除上一轮产出的文件，在predicted 的时候不做clean
        output_dir = self.output_path

        # check output dir exists
        if not os.path.exists(output_dir):
            print('无output文件，无法加载以训练模型')
            os.mkdir(output_dir)
            raise FileNotFoundError('无output文件，无法加载以训练模型')

        print('初始化标签-ID字典等等')
        CONFIG.init(FLAGS.root_dir)

        session_config = tf.ConfigProto(
            # 是否打印使用设备的记录
            log_device_placement=False,
            inter_op_parallelism_threads=0,
            intra_op_parallelism_threads=0,
            # 是否允许自行使用合适的设备
            allow_soft_placement=True)
        run_config = tf.estimator.RunConfig(
            model_dir=output_dir,
            save_summary_steps=500,
            save_checkpoints_steps=120,
            session_config=session_config
        )
        params = {
            'hidden_units': FLAGS.hidden_units,
            'num_layers': FLAGS.num_layers,
            'max_sequence_length': FLAGS.max_sequence_length,
            'dropout_rate': FLAGS.dropout_rate,
            'learning_rate': FLAGS.learning_rate,
        }

        print('构造estimator')
        self.estimator = tf.estimator.Estimator(MODEL.model_fn, config=run_config, params=params)

    def __predict__(self,sentences_words_tags_posTags):
        FLAGS = self.FLAGS
        pre_inf = functools.partial(INPUT.input_fn, input_dir=None, sentences_words_posTags=sentences_words_tags_posTags,
                                    shuffe=False, num_epochs=1, batch_size=FLAGS.batch_size,
                                    max_sequence_length=FLAGS.max_sequence_length)
        predictions = self.estimator.predict(input_fn=pre_inf)
        predictions = [x['pre_ids'] for x in list(predictions)]

        words_list = [one_sentence_words_posTags[0] for one_sentence_words_posTags in sentences_words_tags_posTags]
        tags_list = []
        for pre_ids in predictions:
            tags_list.append([CONFIG.ID_2_TAG[id] for id in pre_ids])
        return words_list,tags_list

    '''抽取方式分为从自然语段中抽取和从已经分好词的词组中抽取'''

    '''从自然语段抽取'''
    def extractor(self,paragraph):
        '''传入原文段，抽取事件，此时需要知道事件中每个词语在原文句中的索引位置，以及原文句'''
        #分句 并判断每句是否含触发词
        sentenceSplitter = SentenceSplitter()
        sentences = []
        for sentence in sentenceSplitter.split(paragraph):
            hasTrigger = self.__ifContainTrigger__(sentence)
            if (not hasTrigger):
                print('该句子中无关注事实：' + sentence)
            else:
                sentences.append(sentence)
        if (len(sentences) == 0):
            print("整个抽取文本无关注事实")
            return []
        words_list, tags_list, words_in_sentence_index_list,sentences = self.__extractorFromSentences__(sentences)
        events = self.__formEvents__(words_list,tags_list)
        return events

    # 判断是否含有关注事实触发词
    def __ifContainTrigger__(self, sentence):
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
    def __extractorFromSentences__(self, sentences):
        sentences_words_posTags = []
        words_in_sentence_index_list = []
        newSentences = []
        for sentence in sentences:
            # 分词、获取pos标签、去停用词
            words = CONFIG.SEGMENTOR.segment(sentence)
            # 每个词对应的在原句里面的索引[beginIndex,endIndex)
            indexPairs = self.__formIndexs__(words)
            postags = CONFIG.POSTAGGER.postag(words)
            tags = ['O' for _ in words]

            # 标记触发词
            triggers = INPUT.findTrigger(sentence)
            if (triggers == None or len(triggers) == 0):
                continue
            for tag, beginIndex, endIndex in triggers:
                words, tags = INPUT.labelTrigger(list(words), tags, beginIndex, endIndex, tag)

            # 去停用词
            newWords = []
            newPosTags = []
            newTags = []
            newIndexs = []
            for word, pos, tag, indexPair in zip(words, postags, tags, indexPairs):
                if (word not in CONFIG.STOP_WORDS):
                    newWords.append(word)
                    newPosTags.append(pos)
                    newTags.append(tag)
                    newIndexs.append(copy.copy(indexPair))
            sentences_words_posTags.append([newWords, newTags, newPosTags])
            words_in_sentence_index_list.append(newIndexs)
            newSentences.append(copy.copy(sentence))

        words_list,tags_list = self.__predict__(sentences_words_posTags)

        return [words_list, tags_list, words_in_sentence_index_list,newSentences]
    def __formIndexs__(self,words):
        indexs = []
        baseIndex = 0
        for word in words:
            indexs.append([baseIndex, baseIndex + len(word)])
            baseIndex += len(word)
        return indexs
    def __formEvents__(self,words_list,tags_list):
        '''传入分词list,tags list,原句索引对list，原句list。构造出所有的抽取到的事件'''
        events = []
        for words, tags in zip(words_list,tags_list):
            event = self.__get_event_from_one_words__(words,tags)
            if(event):
                events.append(event)
        return events

    def __get_event_from_one_words__(self,words,tags):
        # 获取触发词tag
        return EventFactory2(words,tags)


    '''从已经分好词的词组中抽取'''
    def extractor_from_words_posTags(self, words_trigger_tags_pos_tags_list):
        '''传入分词、触发词标签以及pos标签，抽取事实，此时不需要原sentence和index_pair,而且返回的list 每个item对应一句words的抽取事件结果，可能有多个事件'''
        pre_words_list, pre_tags_list = self.__predict__(words_trigger_tags_pos_tags_list)
        events = []
        for pre_words,pre_tags in zip(pre_words_list,pre_tags_list):
            events.append(EventFactory2(pre_words,pre_tags))
        return events

    def release(self):
        CONFIG.release()

if __name__ == '__main__':
    FLAGS = getParser()
    extractor = Event_Detection(FLAGS,output_path='output_15_64_base')
    events = extractor.extractor('原告与被告于2015年底经人介绍相识并确立恋爱关系。原、被告于2007年11月于网上相识恋爱。1997年经人介绍原、被告相识结婚。原告李××诉称，原、被告经人介绍认识订婚，1997年12月10日在汶上县民政局登记结婚，婚后生育一男一女，男孩叫刘×锦，女孩叫刘×华。婚后生育四个女儿，长女王雪斌，次女王雪玲，三女王乙，四女王丙。2017年6月原被告发生争吵后，被告出手殴打了原告，原告回娘家居住，双方分居至今。2008年底，原告确感无法继续与被告生活，外出打工与被告分居至今，并分别于2011年2月、2011年10月向法院提起离婚诉讼，第一次因原告未到庭参加诉讼，法院裁定按撤诉处理，第二次法院判决不准予离婚。2002年原告根据有关政策将被告及孩子户口迁至本市，但仍分居，无奈原告于2007年和2008年先后两次向法院提出离婚请求。婚后因被告脾气暴躁，多次对原告实施家庭暴力，并且还有打牌赌博的恶习，导致夫妻感情完全破裂。请求法院根据双方提交的证据依法分割共同财产，包括双方银行存款、住房公积金、不动产等。')
    print('end')
    sys.exit(0)