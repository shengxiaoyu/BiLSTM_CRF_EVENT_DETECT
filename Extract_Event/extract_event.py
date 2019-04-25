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

    def __init__(self,FLAGS):
        self.FLAGS = FLAGS
        print(FLAGS)
        self.__initFirstModel__()

    def __initFirstModel__(self):
        FLAGS = self.FLAGS
        tf.enable_eager_execution()
        # 配置哪块gpu可见
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device_map

        # 在re train 的时候，才删除上一轮产出的文件，在predicted 的时候不做clean
        output_dir = os.path.join(FLAGS.root_dir, 'output_' + FLAGS.sentence_mode+'_no_pos')

        # check output dir exists
        if not os.path.exists(output_dir):
            print('无output文件，无法加载以训练模型')
            os.mkdir(output_dir)

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

    def formIndexs(self,words):
        indexs = []
        baseIndex = 0
        for word in words:
            indexs.append([baseIndex, baseIndex + len(word)])
            baseIndex += len(word)
        return indexs

    def predict(self,sentences):
        FLAGS = self.FLAGS
        sentences_words_posTags = []
        words_in_sentence_index_list = []
        for sentence in sentences:
            # 分词、获取pos标签、去停用词
            words = CONFIG.SEGMENTOR.segment(sentence)
            # 每个词对应的在原句里面的索引[beginIndex,endIndex)
            indexPairs = self.formIndexs(words)
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
        pre_inf = functools.partial(INPUT.input_fn, input_dir=None, sentences_words_posTags=sentences_words_posTags,
                                    shuffe=False, num_epochs=1, batch_size=FLAGS.batch_size,
                                    max_sequence_length=FLAGS.max_sequence_length)
        predictions = self.estimator.predict(input_fn=pre_inf)
        predictions = [x['pre_ids'] for x in list(predictions)]

        words_list = [one_sentence_words_posTags[0] for one_sentence_words_posTags in sentences_words_posTags]
        tags_list = []
        for pre_ids in predictions:
            tags_list.append([CONFIG.ID_2_TAG[id] for id in pre_ids])
        for words, tags in zip(words_list, tags_list):
            print(' '.join(words))
            print('\n')
            print(' '.join(tags))
            print('\n')
        return [words_list, tags_list, words_in_sentence_index_list]

    # 判断是否含有关注事实触发词
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
        #分句 并判断每句是否含触发词
        sentenceSplitter = SentenceSplitter()
        sentences = []
        for sentence in sentenceSplitter.split(paragraph):
            hasTrigger = self.ifContainTrigger(sentence)
            if (not hasTrigger):
                print('该句子中无关注事实：' + sentence)
            else:
                sentences.append(sentence)
        if (len(sentences) == 0):
            print("整个抽取文本无关注事实")
            return []
        words_list, tags_list, words_in_sentence_index_list = self.predict(sentences)
        events = []
        # 获取触发词tag
        triggers = CONFIG.TRIGGER_TAGs

        for words, tags,words_in_sentence_index_pair,sentence in zip(words_list,tags_list,words_in_sentence_index_list,sentences):
            for index, tag in enumerate(tags):
                if(tag in triggers and tag.find('B_')!=-1):
                    '''发现触发词'''
                    type = tag[2:]
                    completeTrigger = words[index]
                    sentence_char_index_pair = words_in_sentence_index_pair[index]
                    tag_index_pair = [index,index]
                    for endIndex in range(index+1,len(tags)):
                        if(tags[endIndex]=='I_'+type):
                            completeTrigger += words[endIndex]
                            sentence_char_index_pair[1] = words_in_sentence_index_pair[endIndex][1]
                            tag_index_pair[1] += 1
                        else:break

                    event = EventModel.EventFactory(type,completeTrigger,tag_index_pair,sentence,sentence_char_index_pair,words,tags)
                    event.fitArgument(words,tags,words_in_sentence_index_pair)
                    events.append(event)
        return events


    def release(self):
        CONFIG.release()

if __name__ == '__main__':
    FLAGS = getParser()
    FLAGS.ifTrain = False
    FLAGS.ifTest = False
    FLAGS.ifPredict = True
    extractor = Event_Detection(FLAGS)
    events = extractor.extractor('原、被告双方1986年上半年经人介绍认识，××××年××月××日在临桂县宛田乡政府登记结婚，'
                                 '××××年××月××日生育女儿李某乙，××××年××月××日生育儿子李某丙，'
                                 '现女儿李某乙、儿子李某丙都已独立生活')

    print('\n'.join(list(map(lambda x:str(x.__dict__),events))))
    print('end')
    sys.exit(0)