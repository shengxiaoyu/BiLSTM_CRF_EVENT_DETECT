#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import functools
import os
import sys

__doc__ = 'description'
__author__ = '13314409603@163.com'
from pyltp import SentenceSplitter

import First_For_Commo_Tags.config_center as CONFIG
import First_For_Commo_Tags.word2vec_lstm_crf_ed as first
import Second_For_Fine_Tags.word2vec_lstm_crf_argu_match as second
from Config.config_parser import getParser
import Extract_Event.EventModel as EventModel
import Second_For_Fine_Tags.config_center as NEW_CONFIG
from Extract_Event.EventModel import EventFactory2
import tensorflow as tf
from First_For_Commo_Tags import model_fn as FirstModel
from First_For_Commo_Tags import input_fn as First_Input
from Second_For_Fine_Tags import model_fn as SecondModel
from Second_For_Fine_Tags import input_fn as Second_Input

root_dir = 'C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF'

#判断是否含有关注事实触发词



class Event_Detection(object):
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

        #处理多个事件共享触发词


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

    def formEvents(self,words_list,tags_list,words_in_sentence_index_list,sentences):
        '''传入分词list,tags list,原句索引对list，原句list。构造出所有的抽取到的事件'''
        events = []
        for words, tags,words_in_sentence_index_pair,sentence in zip(words_list,tags_list,words_in_sentence_index_list,sentences):
            events.extend(self.__get_event_from_one_words__(words,tags,words_in_sentence_index_pair,sentence))
        return events

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

    def extractor_from_words_posTags(self, words_trigger_tags_pos_tags_list):
        '''传入分词、触发词标签以及pos标签，抽取事实，此时不需要原sentence和index_pair,而且返回的list 每个item对应一句words的抽取事件结果，可能有多个事件'''
        pre_words_list, pre_tags_list = self.__predict__(words_trigger_tags_pos_tags_list)
        events = []
        for pre_words,pre_tags in zip(pre_words_list,pre_tags_list):
            events.append(EventFactory2(pre_words,pre_tags))
        return events

    def release(self):
        CONFIG.release()

class Event_Detection2(object):
    '''通用方法'''
    def __init__(self,FLAGS,first_output_path=None,second_output_path=None):
        self.FLAGS = FLAGS
        print(FLAGS)
        self.sentenceSplitter = SentenceSplitter()
        self.first_output_path = os.path.join(self.FLAGS.root_dir,first_output_path)
        self.__initFirstModel__()
        self.second_output_path = os.path.join(self.FLAGS.root_dir,second_output_path)
        self.__initSecondModel__()

    def __initFirstModel__(self):
        FLAGS = self.FLAGS
        print(FLAGS)

        tf.enable_eager_execution()
        # 配置哪块gpu可见
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device_map

        # 在re train 的时候，才删除上一轮产出的文件，在predicted 的时候不做clean
        output_dir = self.first_output_path

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
        self.first_estimator = tf.estimator.Estimator(FirstModel.model_fn, config=run_config, params=params)

    def __initSecondModel__(self):
        FLAGS = self.FLAGS
        print(FLAGS)

        tf.enable_eager_execution()
        # 配置哪块gpu可见
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device_map

        # 在re train 的时候，才删除上一轮产出的文件，在predicted 的时候不做clean
        output_dir = self.second_output_path
        if not os.path.exists(output_dir):
            print('无output文件，无法加载以训练模型')
            os.mkdir(output_dir)
            raise FileNotFoundError('无output文件，无法加载以训练模型')

        print('初始化标签-ID字典等等')
        NEW_CONFIG.initNewTags(os.path.join(FLAGS.root_dir, 'full_trigger_labels.txt'),
                               os.path.join(FLAGS.root_dir, 'full_argu_labels.txt'))

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
        self.second_estimator = tf.estimator.Estimator(SecondModel.model_fn, config=run_config, params=params)

    def __predict__(self,sentences_words_tags_posTags):
        #第一层模型预测
        FLAGS = self.FLAGS
        pre_inf = functools.partial(First_Input.input_fn, input_dir=None,
                                    sentences_words_posTags=sentences_words_tags_posTags,
                                    shuffe=False, num_epochs=1, batch_size=FLAGS.batch_size,
                                    max_sequence_length=FLAGS.max_sequence_length)
        first_predictions = self.first_estimator.predict(input_fn=pre_inf)
        first_ids_list = [x['pre_ids'] for x in list(first_predictions)]

        first_tags_list = []
        for pre_ids in first_ids_list:
            first_tags_list.append([CONFIG.ID_2_TAG[id] for id in pre_ids])


        #根据第一层预测结果，构造第二层模型预测输入

        def handlerOneInput(words, first_tags,index_of_words):
            results = []
            for index, tag in enumerate(first_tags):
                if (tag in CONFIG.TRIGGER_TAGs and tag.find('B_') != -1):  # 触发词
                    # 不含B_的触发词
                    currentTrigger = tag[2:]
                    # 确定触发词的长度
                    endIndex = index + 1
                    while (first_tags[endIndex].find(currentTrigger) != -1):
                        endIndex += 1
                    # 构造新的tags列：
                    newTags = [first_tags[i] + '_Trigger' if i >= index and i < endIndex else 'O' for i in
                               range(len(first_tags))]
                    # 深拷贝
                    newWords = [x for x in words]
                    results.append([newWords, first_tags, newTags,index_of_words])
            return results



        # 构造第二个模型的输入list
        # 针对第一个模型的预测结果，针对每个触发词都会构成一条新的预测example
        sentence_words_firstTags_trueTriggerTags = []
        index_of_words = 0
        for words_tags_postag, first_tags in zip(sentences_words_tags_posTags, first_tags_list):
            the_words_firstTags_newTags_list = handlerOneInput(words_tags_postag[0], first_tags,index_of_words)
            sentence_words_firstTags_trueTriggerTags.extend(the_words_firstTags_newTags_list)
            index_of_words += 1

        #第二层模型预测
        pred_inpf = functools.partial(Second_Input.input_fn, input_dir=None, shuffe=False, num_epochs=FLAGS.num_epochs,
                                      batch_size=FLAGS.batch_size, max_sequence_length=FLAGS.max_sequence_length,
                                      sentence_words_firstTags_trueTriggerTags=sentence_words_firstTags_trueTriggerTags)
        predictions = self.second_estimator.predict(input_fn=pred_inpf)
        preds = [x['pre_ids'] for x in list(predictions)]

        new_words_list = []
        new_tags_list = []
        words_index_list = []
        for ids, inputs in zip(preds, sentence_words_firstTags_trueTriggerTags):
            # 词语
            words = inputs[0]

            new_words_list.append(words)
            # 预测标签
            tags = [NEW_CONFIG.NEW_ID_2_TAG[id] for id in ids]
            # 每个词语在原句中的index_pair
            new_tags_list.append(tags)
            words_index_list.append(inputs[3])
        return new_words_list, new_tags_list,words_index_list

    '''两中抽取方式，1、从自然语段抽取；2、从分好词的词组中抽取'''

    '''从分好词的词组抽取'''
    def extractor_from_words_posTags(self,words_trigger_tags_pos_tags_list):
        '''传入分词、触发词标签以及pos标签，抽取事实，此时不需要原sentence和index_pair,而且返回的list 每个item对应一句words的抽取事件结果，可能有多个事件'''
        pre_words_list, pre_tags_list ,words_index_list = self.__predict__(words_trigger_tags_pos_tags_list)
        events = []
        current_words_index = words_index_list[0]
        current_events = []
        for pre_words, pre_tags,words_index in zip(pre_words_list, pre_tags_list,words_index_list):
            if(words_index==current_words_index):
                current_events.append(EventFactory2(pre_words, pre_tags))
            else:
                events.append(current_events)
                current_events= []
                current_events.append(EventFactory2(pre_words, pre_tags))
                current_words_index = current_words_index+1
        events.append(current_events)
        return events

    '''从自然语段中抽取'''
    def extractor(self,paragraph):
        # 调用预测接口
        sentences = []
        for sentence in self.sentenceSplitter.split(paragraph):
            hasTrigger = self.__ifContainTrigger__(sentence)
            if (not hasTrigger):
                print('该句子中无关注事实：' + sentence)
            else:
                sentences.append(sentence)
        if (len(sentences) == 0):
            print("整个抽取文本无关注事实")
            return []

        first_words_list = []
        first_tags_list = []
        if(self.first_estimator):
            print('第一个模型预测')
            sentences_words_posTags = []
            for sentence in sentences:
                # 分词、获取pos标签、去停用词
                words = CONFIG.SEGMENTOR.segment(sentence)
                postags = CONFIG.POSTAGGER.postag(words)
                tags = ['O' for _ in words]

                # 标记触发词
                triggers = First_Input.findTrigger(sentence)
                if (triggers == None or len(triggers) == 0):
                    continue
                for tag, beginIndex, endIndex in triggers:
                    words, tags = First_Input.labelTrigger(words, tags, beginIndex, endIndex, tag)

                #去停用词
                newWords = []
                newPosTags = []
                newTags = []
                for word, pos,tag in zip(words, postags,tags):
                    if (word not in CONFIG.STOP_WORDS):
                        newWords.append(word)
                        newPosTags.append(pos)
                        newTags.append(tag)
                sentences_words_posTags.append([newWords,newTags,newPosTags])
            pre_inf = functools.partial(First_Input.input_fn, input_dir=None, sentences_words_posTags=sentences_words_posTags,
                                        shuffe=False, num_epochs=1, batch_size=self.FLAGS.batch_size,
                                        max_sequence_length=self.FLAGS.max_sequence_length)
            predictions = self.first_estimator.predict(input_fn=pre_inf)
            predictions = [x['pre_ids'] for x in list(predictions)]

            first_words_list = [one_sentence_words_posTags[0] for one_sentence_words_posTags in sentences_words_posTags]
            for pre_ids in predictions:
                first_tags_list.append([CONFIG.ID_2_TAG[id]for id in pre_ids])


        # 处理多个事件共享触发词


        second_word_list = []
        second_tags_list = []
        if(self.second_estimator):
            print('第二个模型预测')
            '''根据原句分词和第一个模型的预测标记序列 让第二个模型预测并抽取事实'''
            '''传入的sentence_words_firstTags_list 包括原文分词和第一个模型的预测标签序列'''

            def handlerOneInput(words, first_tags):
                results = []
                for index, tag in enumerate(first_tags):
                    if (tag in CONFIG.TRIGGER_TAGs and tag.find('B_') != -1):  # 触发词
                        # 不含B_的触发词
                        currentTrigger = tag[2:]
                        # 确定触发词的长度
                        endIndex = index + 1
                        while (endIndex < len(first_tags) and first_tags[endIndex].find(currentTrigger) != -1):
                            endIndex += 1
                        # 构造新的tags列：
                        newTags = [first_tags[i] + '_Trigger' if i >= index and i < endIndex else 'O' for i in
                                   range(len(first_tags))]
                        # 深拷贝其余两列
                        newWords = [x for x in words]
                        newFirstTags = [x for x in first_tags]

                        results.append([newWords, newFirstTags, newTags])
                return results

            # 构造第二个模型的输入list
            # 针对第一个模型的预测结果，针对每个触发词都会构成一条新的预测example
            sentence_words_firstTags_trueTriggerTags = []
            for words, first_tags in zip(first_words_list, first_tags_list):
                the_words_firstTags_newTags_list = handlerOneInput(words, first_tags)
                sentence_words_firstTags_trueTriggerTags.extend(the_words_firstTags_newTags_list)
            pred_inpf = functools.partial(Second_Input.input_fn, input_dir=None, shuffe=False, num_epochs=self.FLAGS.num_epochs,
                                          batch_size=self.FLAGS.batch_size, max_sequence_length=self.FLAGS.max_sequence_length,
                                          sentence_words_firstTags_trueTriggerTags=sentence_words_firstTags_trueTriggerTags)
            predictions = self.second_estimator.predict(input_fn=pred_inpf)
            preds = [x['pre_ids'] for x in list(predictions)]

            for ids, inputs in zip(preds, sentence_words_firstTags_trueTriggerTags):
                # 词语
                words = inputs[0]
                second_word_list.append(words)
                # 预测标签
                tags = [NEW_CONFIG.NEW_ID_2_TAG[id] for id in ids]
                # 每个词语在原句中的index_pair
                second_tags_list.append(tags)

        events = []
        for tags, words in zip(second_tags_list, second_word_list):
            events.append(EventFactory2(words,tags))
        return events

    def __ifContainTrigger__(self,sentence):
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

    def release(self):
        CONFIG.release()

def shareBear(tags):
    bearCount = 0
    nameCount = 0
    ageCount = 0
    genderCount = 0
    for tag in tags:
        if(tag=='B_Bear'):
            bearCount+=1
        if(tag=='B_Name'):
            nameCount+=1
        if(tag=='B_Age'):
            ageCount+=1
        if(tag=='B_Gender'):
            genderCount += 1
    if(nameCount>bearCount or ageCount>bearCount or genderCount>bearCount):
         pass
if __name__ == '__main__':
    extractor = Event_Detection2(getParser(),first_output_path='output_15_64_pos_trigger_Merge',second_output_path='second_output_15_64_Trigger')
    events = extractor.extractor('原告李××诉称，原、被告经人介绍认识订婚，1997年12月10日在汶上县民政局登记结婚，婚后生育一男一女，男孩叫刘×锦，女孩叫刘×华'
                                       '婚后生育四个女儿，长女王雪斌，次女王雪玲，三女王乙，四女王丙。'
                                 )
    print('end')
    exit(0)