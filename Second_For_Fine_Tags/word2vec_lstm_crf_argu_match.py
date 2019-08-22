#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__doc__ = 'description'
__author__ = '13314409603@163.com'

import functools
import os

import numpy as np
import tensorflow as tf
from sklearn_crfsuite.metrics import flat_classification_report
import First_For_Commo_Tags.config_center as CONFIG
import Second_For_Fine_Tags.input_fn as INPUT
import Second_For_Fine_Tags.model_fn as MODEL
import Second_For_Fine_Tags.config_center as NEW_CONFIG

#训练、评估、预测,sentencs_words_firstTags_list:要预测的句子+第一层模型初步预测结果
def main(FLAGS,sentencs_words_firstTags_list=None,words_firstTags_indxPairs_sentences=None):
    print(FLAGS)

    tf.enable_eager_execution()
    # 配置哪块gpu可见
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device_map

    # 在re train 的时候，才删除上一轮产出的文件，在predicted 的时候不做clean
    output_dir = os.path.join(FLAGS.root_dir,'second_output_'+str(FLAGS.num_epochs)+'_'+str(FLAGS.batch_size)+'_'+FLAGS.sentence_mode)
    if FLAGS.ifTrain:
        if os.path.exists(output_dir):
            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                    else:
                        os.remove(c_path)

            try:
                print('清除历史训练记录')
                del_file(output_dir)
            except Exception as e:
                print(e)
                print('pleace remove the files of output dir and data.conf')
                exit(-1)
    # check output dir exists
    if not os.path.exists(output_dir):
        print('创建output文件夹')
        os.mkdir(output_dir)


    print('初始化标签-ID字典等等')
    CONFIG.init(FLAGS.root_dir)
    NEW_CONFIG.initNewTags(os.path.join(FLAGS.root_dir,'full_trigger_labels.txt'),os.path.join(FLAGS.root_dir,'full_argu_labels.txt'))


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
        'hidden_units':FLAGS.hidden_units,
        'num_layers':FLAGS.num_layers ,
        'max_sequence_length':FLAGS.max_sequence_length,
        'dropout_rate':FLAGS.dropout_rate,
        'learning_rate':FLAGS.learning_rate,
    }

    print('构造estimator')
    estimator = tf.estimator.Estimator(MODEL.model_fn,config=run_config,params=params)
    # estimator
    if FLAGS.ifTrain :
        print('获取训练数据。。。')
        train_inpf = functools.partial(INPUT.input_fn, input_dir=(os.path.join(FLAGS.labeled_data_path+'_for_second', 'train')),
                                       shuffe=True, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size,max_sequence_length=FLAGS.max_sequence_length)
        train_examples = list(train_inpf())
        train_total = len(train_examples)
        print('训练steps:'+str(train_total))
        print('获取评估数据。。。')
        eval_inpf = functools.partial(INPUT.input_fn, input_dir=(os.path.join(FLAGS.labeled_data_path+'_for_second', 'dev')),
                                      shuffe=False, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size,max_sequence_length=FLAGS.max_sequence_length)
        hook = tf.contrib.estimator.stop_if_no_increase_hook(
            estimator, 'f1', 500, min_steps=8000, run_every_secs=120)
        dev_total = len(list(eval_inpf()))
        print('评估总数：' + str(dev_total))
        train_spec = tf.estimator.TrainSpec(input_fn=train_inpf,hooks=[hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf)
        print('开始训练+评估。。。')
        tf.estimator.train_and_evaluate(estimator,train_spec,eval_spec)

    if FLAGS.ifTest:
        print('获取预测数据。。。')
        test_inpf = functools.partial(INPUT.input_fn, input_dir=(os.path.join(FLAGS.labeled_data_path+'_for_second', 'test')),
                                      shuffe=False, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size,max_sequence_length=FLAGS.max_sequence_length)

        predictions = estimator.predict(input_fn=test_inpf)
        pred_true = INPUT.generator_fn(input_dir=(os.path.join(FLAGS.labeled_data_path+'_for_second', 'test')),max_sequence_length = FLAGS.max_sequence_length,noEmbedding=True)

        #取真实的tags
        targets = [x[1] for x in pred_true]
        print('预测总数：' + str(len(targets)))
        #预测结果
        pred = [x['pre_ids'] for x in list(predictions)]
        #预测分析
        #真实tag id 除了<pad>和O
        indices = [item[1] for item in CONFIG.TAG_2_ID.items() if (item[0] != '<pad>' and item[0] != 'O')]
        report = flat_classification_report(y_pred=pred,y_true=targets,labels=indices)
        print(report)

        with open(os.path.join(output_dir,'predict_result.txt'),'w',encoding='utf8') as fw:
            fw.write(str(report))
            for target,predict in zip(pred_true,pred):
                (words,length,first_tags,_),tags = target
                words = [words[i] for i in range(length)]
                first_tags = [first_tags[i] for i in range(length)]
                labels = [NEW_CONFIG.NEW_ID_2_TAG[tags[i]] for i in range(length)]
                outputs = [NEW_CONFIG.NEW_ID_2_TAG[predict[i]] for i in range(length)]
                fw.write('原 文 ：'+' '.join(words))
                fw.write('\n')
                fw.write('第一层预测标签：'+' '.join(first_tags))
                fw.write('\n')
                fw.write('人工标记： '+' '.join(labels))
                fw.write('\n')
                fw.write('预测结果： '+' '.join(outputs))
                fw.write('\n')
                fw.write('\n')
    if FLAGS.ifPredict and sentencs_words_firstTags_list:
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
                    while (endIndex<len(first_tags) and first_tags[endIndex].find(currentTrigger) != -1):
                        endIndex += 1
                    # 构造新的tags列：
                    newTags = [first_tags[i] + '_Trigger' if i >= index and i < endIndex else 'O' for i in
                               range(len(first_tags))]
                    # 深拷贝其余两列
                    newWords = [x for x in words]
                    newFirstTags = [x for x in first_tags]

                    results.append([newWords, newFirstTags, newTags])
            return results

        words_list, first_tags_list = sentencs_words_firstTags_list
        # 构造第二个模型的输入list
        # 针对第一个模型的预测结果，针对每个触发词都会构成一条新的预测example
        sentence_words_firstTags_trueTriggerTags = []
        for words, first_tags in zip(words_list, first_tags_list):
            the_words_firstTags_newTags_list = handlerOneInput(words, first_tags)
            sentence_words_firstTags_trueTriggerTags.extend(the_words_firstTags_newTags_list)
        pred_inpf = functools.partial(INPUT.input_fn, input_dir=None, shuffe=False, num_epochs=FLAGS.num_epochs,
                                      batch_size=FLAGS.batch_size, max_sequence_length=FLAGS.max_sequence_length,
                                      sentence_words_firstTags_trueTriggerTags=sentence_words_firstTags_trueTriggerTags)
        predictions = estimator.predict(input_fn=pred_inpf)
        preds = [x['pre_ids'] for x in list(predictions)]

        new_words_list = []
        new_tags_list = []
        for ids, inputs in zip(preds, sentence_words_firstTags_trueTriggerTags):
            # 词语
            words = inputs[0]
            new_words_list.append(words)
            # 预测标签
            tags = [NEW_CONFIG.NEW_ID_2_TAG[id] for id in ids]
            # 每个词语在原句中的index_pair
            new_tags_list.append(tags)
        for words, tags in zip(new_words_list, new_tags_list):
            print(' '.join(words))
            print('\n')
            print(' '.join(tags))
            print('\n')
        return new_words_list, new_tags_list

    if(FLAGS.ifPredict and words_firstTags_indxPairs_sentences):
        '''根据原句分词和第一个模型的预测标记序列 让第二个模型预测并抽取事实'''
        '''传入的sentence_words_firstTags_list 包括原文分词和第一个模型的预测标签序列,以及每个原文分词在原句中的起止索引'''

        def handlerOneInput(words, first_tags, index_pairs, sentence):
            results = []
            index_pairs_list = []
            sentences = []
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
                    # 深拷贝其余两列
                    newWords = [x for x in words]
                    new_index_pairs = [x for x in index_pairs]
                    results.append([newWords, first_tags, newTags])
                    index_pairs_list.append(new_index_pairs)
                    sentences.append(sentence)
            return results, index_pairs_list, sentences

        words_list, first_tags_list, index_pairs_list, sentences = words_firstTags_indxPairs_sentences
        # 构造第二个模型的输入list
        # 针对第一个模型的预测结果，针对每个触发词都会构成一条新的预测example
        sentence_words_firstTags_trueTriggerTags = []
        new_index_pairs_list = []
        new_sentences = []
        for words, first_tags, index_pairs, sentence in zip(words_list, first_tags_list, index_pairs_list, sentences):
            the_words_firstTags_newTags_list, the_index_pairs_list, the_sentences = handlerOneInput(words, first_tags,
                                                                                                    index_pairs,
                                                                                                    sentence)
            sentence_words_firstTags_trueTriggerTags.extend(the_words_firstTags_newTags_list)
            new_index_pairs_list.extend(the_index_pairs_list)
            new_sentences.extend(the_sentences)
        pred_inpf = functools.partial(INPUT.input_fn, input_dir=None, shuffe=False, num_epochs=FLAGS.num_epochs,
                                      batch_size=FLAGS.batch_size, max_sequence_length=FLAGS.max_sequence_length,
                                      sentence_words_firstTags_trueTriggerTags=sentence_words_firstTags_trueTriggerTags)
        predictions = estimator.predict(input_fn=pred_inpf)
        preds = [x['pre_ids'] for x in list(predictions)]

        new_words_list = []
        new_tags_list = []
        for ids, inputs in zip(preds, sentence_words_firstTags_trueTriggerTags):
            # 词语
            words = inputs[0]
            new_words_list.append(words)
            # 预测标签
            tags = [NEW_CONFIG.NEW_ID_2_TAG[id] for id in ids]
            # 每个词语在原句中的index_pair
            new_tags_list.append(tags)
        for words, tags in zip(new_words_list, new_tags_list):
            print(' '.join(words))
            print('\n')
            print(' '.join(tags))
            print('\n')
        return new_words_list, new_tags_list, new_index_pairs_list, new_sentences

if __name__ == '__main__':
    a = [[1,2,3],[4,5,6]]
    b = [[7],[8]]
    a = np.array(a)
    b = np.array(b)
    c = tf.concat([a,b],axis=-1)
    print(c)
    pass