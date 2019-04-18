#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import functools

from Config.config_parser import getParser

__doc__ = 'description'
__author__ = '13314409603@163.com'
from pyltp import SentenceSplitter

import LSTM_CRF.config_center as CONFIG
import Extract_Event.EventModel as EventModel
import tensorflow as tf
import os
import LSTM_CRF.model_fn as MODEL
import LSTM_CRF.input_fn as INPUT

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
        output_dir = os.path.join(FLAGS.root_dir, 'output_' + FLAGS.sentence_mode)

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

    def predict(self,sentences):
        sentences_words_posTags = []
        for sentence in sentences:
            # 分词、获取pos标签、去停用词
            words = CONFIG.SEGMENTOR.segment(sentence)
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
            for word, pos, tag in zip(words, postags, tags):
                if (word not in CONFIG.STOP_WORDS):
                    newWords.append(word)
                    newPosTags.append(pos)
                    newTags.append(tag)
            sentences_words_posTags.append([newWords, newTags, newPosTags])
        pre_inf = functools.partial(INPUT.input_fn, input_dir=None, sentences_words_posTags=sentences_words_posTags,
                                    shuffe=False, num_epochs=1, batch_size=self.FLAGS.batch_size,
                                    max_sequence_length=self.FLAGS.max_sequence_length)
        predictions = self.estimator.predict(input_fn=pre_inf)
        predictions = [x['pre_ids'] for x in list(predictions)]

        result = []
        for one_sentence_words_posTags, pre_ids in zip(sentences_words_posTags, predictions):
            words = one_sentence_words_posTags[0]
            pre_tags = [CONFIG.ID_2_TAG[id] for id in pre_ids]
            result.append([words, pre_tags])
            print(' '.join(words))
            print(' '.join(pre_tags[0:len(words)]))
        return result

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
            if (len(triggerType) == 0):
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
        predictions = self.predict(sentences)
        events = []
        # 获取触发词tag
        triggers = CONFIG.TRIGGER_TAGs

        for words, tags in predictions:
            hasBegin = False
            currentTraigger = None
            beginIndex = 0
            endIndex = 0
            for index, word in enumerate(tags):
                if (word in triggers):  # 如果是触发词
                    if (word.find('B_') != -1):  # 如果是B_开头
                        if (hasBegin):  # 如果前面有触发词还在统计
                            event = self.saveOneEvent(currentTraigger, beginIndex, endIndex, words, tags)
                            events.append(event)
                        # 新起一个事件
                        hasBegin = True
                        currentTraigger = word[2:]
                        beginIndex = index
                        endIndex = index
                    else:  # I_开头
                        if (hasBegin):  # 此时正在一个触发词的查找范围内
                            if (word.find(currentTraigger) != -1):  # 同一个触发词
                                endIndex = index
                            else:  # 此时在找触发词，但是来了个其他触发词的I_
                                event = self.saveOneEvent(currentTraigger, beginIndex, endIndex, words, tags)
                                events.append(event)
                                hasBegin = True
                                currentTraigger = word[2:]
                                beginIndex = index
                                endIndex = index
                        else:  # 此时没有找触发词直接来了个I_
                            hasBegin = True
                            currentTraigger = word[2:]
                            beginIndex = index
                            endIndex = index
                else:
                    if (hasBegin):  # 查找触发词正常结束
                        event = self.saveOneEvent(currentTraigger, beginIndex, endIndex, words, tags)
                        events.append(event)
                        hasBegin = False
                        beginIndex = 0
                        endIndex = 0
        return events

    def saveOneEvent(self,trigger, beginIndex, endIndex, words, tags):
        completeWord = ''
        for word in words[beginIndex:endIndex + 1]:
            completeWord += word
        # 先把前面的事件确定了
        event = EventModel.EventFactory(trigger, completeWord, beginIndex, endIndex)
        event.fitArgument(words, tags)
        return event

    def release(self):
        CONFIG.release()

if __name__ == '__main__':
    FLAGS = getParser()
    FLAGS.ifTrain = False
    FLAGS.ifTest = False
    FLAGS.ifPredict = True
    extractor = Event_Detection(FLAGS)
    events = extractor.extractor('原、被告双方1986年上半年经人介绍认识，××××年××月××日在临桂县宛田乡政府登记结婚，××××年××月××日生育女儿李某乙，××××年××月××日生育儿子李某丙，现女儿李某乙、儿子李某丙都已独立生活')
    print('end')
    pass