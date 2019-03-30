#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'

import tensorflow as tf
import os
import functools
from Word2Vec.my_word2vec import Word2VecModel
import numpy as np


WV = None
TAG_2_ID = {}
ID_2_TAG = {}



def initTagsAndWord2Vec(rootdir):
    global WV,TAG_2_ID,ID_2_TAG
    WV = Word2VecModel(os.path.join(rootdir, 'word2vec'), '', 30).getEmbedded()
    # <pad> -- <pad> fill word2vec and tags
    WV.add('<pad>', np.zeros(WV.vector_size))

    TAG_2_ID['<pad>'] = len(TAG_2_ID)
    ID_2_TAG[len(ID_2_TAG)] = '<pad>'

    #读取根目录下的labelds文件生成tag—id
    with open(os.path.join(rootdir, 'labels.txt'),'r',encoding='utf8') as f:
        index = 1
        isBegin = False
        for line in f.readlines():
            TAG_2_ID[line.strip()] = index
            ID_2_TAG[index] = line.strip()
            index += 1

def paddingAndEmbedding(words,tags,max_sequence_length):

    length = len(words)
    #padding or cutting
    if(length<max_sequence_length):
        for i in range(length,max_sequence_length):
            words.append('<pad>')
            tags.append('<pad>')
    else:
        words = words[:max_sequence_length]
        tags = tags[:max_sequence_length]

    #embedding
    #如果是词汇表中没用的词，则使用<pad>代替
    for index in range(max_sequence_length):
        try:
            WV[words[index]]
        except:
            words[index] = '<pad>'

    words = [WV[word] for word in words]
    tags = [TAG_2_ID[tag] for tag in tags]

    return (words,length),tags

def generator_fn(input_dir,max_sequence_length):
    for input_file in os.listdir(input_dir):
        with open(os.path.join(input_dir,input_file),'r',encoding='utf8') as f:
            sentence = f.readline()#句子行
            while sentence:
                #标记行
                label = f.readline()
                if not label:
                    break
                words = sentence.strip().split(' ')
                words = list(filter(lambda word:word!='',words))

                tags = label.strip().split(' ')
                tags = list(filter(lambda word:word!='',tags))

                sentence = f.readline()

                if (len(words) != len(tags)):
                    print(input_file, ' words和labels数不匹配：' + sentence + ' words length:' + str(
                        len(words)) + ' labels length:' + str(len(tags)))
                    # sentence = f.readline()
                    continue
                yield paddingAndEmbedding(words,tags,max_sequence_length)

def input_fn(input_dir,shuffe,num_epochs,batch_size,max_sequence_length):
    global WV
    shapes = (([max_sequence_length,WV.vector_size],()),[max_sequence_length])
    types = ((tf.float32,tf.int32),tf.int32)
    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn,input_dir=input_dir,max_sequence_length = max_sequence_length),
        output_shapes=shapes,
        output_types=types
    )
    if shuffe:
        dataset = dataset.shuffle(buffer_size=1000).repeat(num_epochs)

    dataset = dataset.batch(batch_size)
    return dataset

def model_fn(features,labels,mode,params):
    is_training = (mode ==  tf.estimator.ModeKeys.TRAIN)
    features,lengths = features

    # LSTM

    #？
    t = tf.transpose(features, perm=[1, 0, 2])
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['hidden_units'])
    # lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['hidden_units'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_fw)
    output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=lengths)
    output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=lengths)
    output = tf.concat([output_fw, output_bw], axis=-1)

    #?
    output = tf.transpose(output, perm=[1, 0, 2])
    output = tf.layers.dropout(output, rate=params['dropout_rate'], training=is_training)
    #activation= softmax?
    logits = tf.layers.dense(output, len(TAG_2_ID))

    # CRF
    crf_params = tf.get_variable("crf", [len(TAG_2_ID), len(TAG_2_ID)], dtype=tf.float32)
    pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, lengths)


    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        predictions = {
            'pre_ids':pred_ids,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Loss
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, labels, lengths, crf_params)
        loss = tf.reduce_mean(-log_likelihood)

        # Metrics
        weights = tf.sequence_mask(lengths,maxlen=params['max_sequence_length'])
        metrics = {
            'acc': tf.metrics.accuracy(labels, pred_ids, weights),
            # 'precision': precision(labels, pred_ids, num_tags, indices, weights),
            # 'recall': recall(labels, pred_ids, num_tags, indices, weights),
            # 'f1': f1(labels, pred_ids, num_tags, indices, weights),
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(
                loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)

def main(FLAGS):

    # tf.enable_eager_execution()
    # 配置哪块gpu可见
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device_map

    # 在re train 的时候，才删除上一轮产出的文件，在predicted 的时候不做clean
    output_dir = os.path.join(FLAGS.root_dir,'output')
    if FLAGS.mode =='train':
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
                del_file(output_dir)
            except Exception as e:
                print(e)
                print('pleace remove the files of output dir and data.conf')
                exit(-1)
    # check output dir exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    initTagsAndWord2Vec(FLAGS.root_dir)

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

    estimator = tf.estimator.Estimator(model_fn,config=run_config,params=params)
    # estimator
    if FLAGS.mode == 'train' :
        train_inpf = functools.partial(input_fn, input_dir=(os.path.join(FLAGS.labeled_data_path, 'train')),
                                       shuffe=True, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size,max_sequence_length=FLAGS.max_sequence_length)
        eval_inpf = functools.partial(input_fn, input_dir=(os.path.join(FLAGS.labeled_data_path, 'dev')),
                                      shuffe=True, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size,max_sequence_length=FLAGS.max_sequence_length)
        hook = tf.contrib.estimator.stop_if_no_increase_hook(
            estimator, 'f1', 500, min_steps=8000, run_every_secs=120)
        train_spec = tf.estimator.TrainSpec(input_fn=train_inpf,hooks=[hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf)
        tf.estimator.train_and_evaluate(estimator,train_spec,eval_spec)
    if FLAGS.mode =='predict':
        test_inpf = functools.partial(input_fn, input_dir=(os.path.join(FLAGS.labeled_data_path, 'test')),
                                      shuffe=False, num_epochs=1, batch_size=FLAGS.batch_size,max_sequence_length=FLAGS.max_sequence_length)
        predictions = estimator.predict(input_fn=test_inpf)
        targets = generator_fn(input_dir=(os.path.join(FLAGS.labeled_data_path, 'test')),max_sequence_length = FLAGS.max_sequence_length)
        predictions = filter(lambda x:x != TAG_2_ID['<pad>'],predictions)
        for target,predict in zip(targets,predictions):
            (_,length),tags = target
            labels = "人工标记："
            output = "预测结果："
            pre_ids = predict['pre_ids']
            # for tag_id,pre_tag in zip(tags,predict['pre_ids']):
            #     labels += ID_2_TAG[tag_id]
            #     output += ID_2_TAG[pre_tag]
            for index in range(length):
                labels += ID_2_TAG[tags[index]]
                output += ID_2_TAG[pre_ids[index]]
            print('\n'.join([labels,output]))
            print('\n')
if __name__ == '__main__':
    # tf.enable_eager_execution()
    # rootdir = 'C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\'
    # initTagsAndWord2Vec(rootdir)
    # data = input_fn('C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\NERdata\\train', True, 1, 5, 50)
    # for v in data:
    #     print(v)
    pass