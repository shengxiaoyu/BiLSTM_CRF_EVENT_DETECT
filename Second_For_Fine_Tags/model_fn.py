#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tf_metrics import precision
from tf_metrics import recall
from tf_metrics import f1

__doc__ = 'description'
__author__ = '13314409603@163.com'
import tensorflow as tf
# from tf_metrics import recall, f1, precision
import First_For_Commo_Tags.config_center as CONFIG
import Second_For_Fine_Tags.config_center as NEW_CONFIG

def model_fn(features,labels,mode,params):
    is_training = (mode ==  tf.estimator.ModeKeys.TRAIN)
    #传入的features: ((句子每个单词向量，句子真实长度），句子每个tag索引).batchSize为5
    features,lengths,oldTags,triggerFeatures = features
    # LSTM
    print('构造LSTM层')

    #转换为lstm时间序列输入
    t = tf.transpose(features, perm=[1, 0, 2])
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['hidden_units'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['hidden_units'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)

    print('LSTM联合层')
    output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=lengths)
    output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=lengths) #shape 49*batch_size*100
    output = tf.concat([output_fw, output_bw], axis=-1) #55*batch_size*200
    #转换回正常输出
    output = tf.transpose(output, perm=[1, 0, 2]) #batch_size*55*200
    print('dropout')
    output = tf.layers.dropout(output, rate=params['dropout_rate'], training=is_training)

    # 添加第一层预测标签特征
    print('添加第一层预测标签特征')
    output = tf.concat([output, oldTags], axis=-1)

    #添加是否是关注触发词 特征
    print('添加是否是关注触发词特征')
    output = tf.concat([output, triggerFeatures], axis=-1)

    #全连接层
    logits = tf.layers.dense(output, NEW_CONFIG.NEW_TAGs_LEN) #batch_size*55*len(tags)


    print('CRF层')
    # CRF

    crf_params = tf.get_variable("crf", [NEW_CONFIG.NEW_TAGs_LEN, NEW_CONFIG.NEW_TAGs_LEN], dtype=tf.float32)
    pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, lengths)


    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        print('预测。。。')
        predictions = {
            'pre_ids':pred_ids,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Loss
        print('loss计算')
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, labels, lengths, crf_params)
        loss = tf.reduce_mean(-log_likelihood)

        if mode == tf.estimator.ModeKeys.EVAL:
            # return None ;
            print('评估。。。')
            # Metrics
            weights = tf.sequence_mask(lengths, maxlen=params['max_sequence_length'])
            indices = [item[1] for item in NEW_CONFIG.NEW_TAG_2_ID.items() if (item[0]!='<pad>'and item[0]!='O')]
            metrics = {
                'acc': tf.metrics.accuracy(labels, pred_ids, weights),
                'precision': precision(labels, pred_ids, NEW_CONFIG.NEW_TAGs_LEN,indices, weights),
                'recall': recall(labels, pred_ids, NEW_CONFIG.NEW_TAGs_LEN,indices,  weights),
                'f1': f1(labels, pred_ids, NEW_CONFIG.NEW_TAGs_LEN,indices, weights),
            }
            for metric_name, op in metrics.items():
                tf.summary.scalar(metric_name, op[1])

            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        else:
            print('训练。。。')
            train_op = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(
                loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

if __name__ == '__main__':
    pass