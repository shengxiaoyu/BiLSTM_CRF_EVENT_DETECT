#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import functools
import sys

from Config.config_parser import getParser

__doc__ = 'description'
__author__ = '13314409603@163.com'
from pyltp import SentenceSplitter

import First_For_Commo_Tags.config_center as CONFIG
import Extract_Event.EventModel as EventModel
import tensorflow as tf
import os
import First_For_Commo_Tags.model_fn as MODEL
import First_For_Commo_Tags.input_fn as INPUT

class Event_Detection(object):

    def __init__(self,FLAGS,output_path=None):
        self.FLAGS = FLAGS
        print(FLAGS)
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


    '''通用方法'''
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

    def __get_event_from_one_words__(self,words,tags,words_in_sentence_index_pairs,sentence):
        # 获取触发词tag
        triggers = CONFIG.TRIGGER_TAGs
        events = []
        for index, tag in enumerate(tags):
            if (tag in triggers and tag.find('B_') != -1):
                '''发现触发词'''
                type = tag[2:]
                completeTrigger = words[index]
                # sentence_char_index_pair = words_in_sentence_index_pairs[index]
                tag_index_pair = [index, index]
                for endIndex in range(index + 1, len(tags)):
                    if (tags[endIndex] == 'I_' + type):
                        completeTrigger += words[endIndex]
                        # sentence_char_index_pair[1] = words_in_sentence_index_pairs[endIndex][1]
                        tag_index_pair[1] += 1
                    else:
                        break
                event = EventModel.EventFactory(type, completeTrigger, tag_index_pair, sentence,
                                                words_in_sentence_index_pairs, words, tags)
                event.fitArgument(words, tags, words_in_sentence_index_pairs)
                events.append(event)
        return events

    '''分别代表从原自然语段中抽取和已经分词获得pos标签之后的词组中抽取'''
    '''从完整的原文段落中抽取事件'''
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
        words_list, tags_list, words_in_sentence_index_list = self.__extractorFromSentences__(sentences)
        events = self.__formEvents__(words_list,tags_list,words_in_sentence_index_list,sentences)
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
                words, tags = INPUT.labelTrigger(words, tags, beginIndex, endIndex, tag)

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
                    newIndexs.append(indexPair)
            sentences_words_posTags.append([newWords, newTags, newPosTags])
            words_in_sentence_index_list.append(newIndexs)
            words_list,tags_list = self.__predict__(sentences_words_posTags)

        return [words_list, tags_list, words_in_sentence_index_list]

    def __formIndexs__(self,words):
        indexs = []
        baseIndex = 0
        for word in words:
            indexs.append([baseIndex, baseIndex + len(word)])
            baseIndex += len(word)
        return indexs

    def __formEvents__(self,words_list,tags_list,words_in_sentence_index_list,sentences):
        '''传入分词list,tags list,原句索引对list，原句list。构造出所有的抽取到的事件'''
        events = []
        for words, tags,words_in_sentence_index_pair,sentence in zip(words_list,tags_list,words_in_sentence_index_list,sentences):
            events.extend(self.__get_event_from_one_words__(words,tags,words_in_sentence_index_pair,sentence))
        return events


    '''从已经分词，去停用词，pos和是否触发词都已经或的数组中抽取事件'''
    def extractor_from_words_posTags(self, words_trigger_tags_pos_tags_list):
        '''传入分词、触发词标签以及pos标签，抽取事实，此时不需要原sentence和index_pair,而且返回的list 每个item对应一句words的抽取事件结果，可能有多个事件'''
        pre_words_list, pre_tags_list = self.__predict__(words_trigger_tags_pos_tags_list)
        events = self.formEvents2(pre_words_list, pre_tags_list)
        return events

    def formEvents2(self,words_list,tags_list):
        '''根据分词和预测的tags构造事件，这是为了用于评估从spe格式中抽取的事件和实际事件。此时没有原句和原句索引，而且每个句子抽出来的事件独立在一个list中'''
        events = []
        for words,tags in zip(words_list,tags_list):
            words_in_sentence_indexs = [[0, 0] for _ in range(len(words))]
            events.append(self.__get_event_from_one_words__(words,tags,words_in_sentence_indexs,''))
        return events


    def release(self):
        CONFIG.release()

if __name__ == '__main__':
    FLAGES = getParser() ;
    extractor = Event_Detection(FLAGES,output_path='output_15_64_pos_trigger_Merge')
    events = extractor.extractor('××××年××月，原、被告经人介绍在家乡举行结婚仪式同居生活，生有一男一女两个孩子。'
                                 ' 原告诉称，原、被告经人介绍，于1987年12月14日登记结婚，婚后生育两个女儿，大女儿已成年，二女儿马某甲现年17周岁。'
                                 '婚续期间，生育三个女孩，长女王某乙1992年2月5日出生，次女王某丙1994年12月21日出生，小女张甲1996年12月16日出生。')
    print('end')
    print('end')
    sys.exit(0)