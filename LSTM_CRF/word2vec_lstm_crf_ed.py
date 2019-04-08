#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pyltp import Segmentor

__doc__ = 'description'
__author__ = '13314409603@163.com'

import tensorflow as tf
from tf_metrics import recall,f1,precision
import os
import functools
from Word2Vec.my_word2vec import Word2VecModel
import numpy as np
from sklearn_crfsuite.metrics import flat_classification_report
import pandas as pd
from pyltp import Postagger

WV = None

TAG_2_ID = {}
ID_2_TAG = {}

# 分词器
SEGMENTOR = Segmentor()
STOP_WORDS=set()



def initTagsAndWord2Vec(rootdir):
    initTags(os.path.join(rootdir,'triggerLabels.txt'),os.path.join(rootdir, 'argumentLabels.txt'))
    initWord2Vec(os.path.join(rootdir, 'word2vec'))
    initPyltpModel(os.path.join(rootdir,'ltp_data_v3.4.0'))
    initStopWords(os.path.join(rootdir, 'newStopWords.txt'))
def initTags(triggerLablePath,argumentLabelPath):
    global TAG_2_ID, ID_2_TAG
    # 把<pad>也加入tag字典
    TAG_2_ID['<pad>'] = len(TAG_2_ID)
    ID_2_TAG[len(ID_2_TAG)] = '<pad>'
    # 读取根目录下的labelds文件生成tag—id
    index = 1
    #获取参数tag
    with open(argumentLabelPath, 'r', encoding='utf8') as f:
        for line in f.readlines():
            TAG_2_ID[line.strip()] = index
            ID_2_TAG[index] = line.strip()
            index += 1
    #获取触发词tag
    with open(triggerLablePath, 'r', encoding='utf8') as f:
        for line in f.readlines():
            TAG_2_ID[line.strip()] = index
            ID_2_TAG[index] = line.strip()
            index += 1
def initWord2Vec(word2vec_model_path):
    global WV
    WV = Word2VecModel(word2vec_model_path, '', 30).getEmbedded()
    # <pad> -- <pad> fill word2vec and tags，添加一个<pad>-向量为0的，用于填充
    WV.add('<pad>', np.zeros(WV.vector_size))
def initPyltpModel(ltp_path):
    global SEGMENTOR
    #初始化模型
    SEGMENTOR.load_with_lexicon(os.path.join(ltp_path,'cws.model'), os.path.join(ltp_path,'userDict.txt'))
def initStopWords(path):
    # 停用词集
    global STOP_WORDS
    with open(path, 'r', encoding='utf8') as f:
        STOP_WORDS = set(f.read().split())

def paddingAndEmbedding(fileName,words,tags,max_sequence_length,noEmbedding):
    # print(fileName)
    length = len(words)

    # 如果是词汇表中没有的词，则使用<pad>代替
    for index in range(length):
        try:
            WV[words[index]]
        except:
            words[index] = '<pad>'


    #padding or cutting
    if(length<max_sequence_length):
        for i in range(length,max_sequence_length):
            words.append('<pad>')
            tags.append('<pad>')
    else:
        words = words[:max_sequence_length]
        tags = tags[:max_sequence_length]


    #根据noEmbedding参数确定是否进行向量化
    if(not noEmbedding):
        words = [WV[word] for word in words]
    try:
        tags = [TAG_2_ID[tag] for tag in tags]
    except:
        print('这个文件tag无法找到正确索引，请检查:'+fileName)

    return (words,min(length,max_sequence_length)),tags
def release():
    SEGMENTOR.release()

def generator_fn(input_dir,max_sequence_length,noEmbedding=False,one_sentence_words=None,):
    result = []
    if(one_sentence_words):
        tags = ['O' for _ in range(len(one_sentence_words))]
        result.append(paddingAndEmbedding('sentence', one_sentence_words, tags, max_sequence_length, noEmbedding))
    elif(input_dir):
        for input_file in os.listdir(input_dir):
            with open(os.path.join(input_dir,input_file),'r',encoding='utf8') as f:
                sentence = f.readline()#句子行
                while sentence:
                    #标记行
                    label = f.readline()
                    pos = f.readline()
                    if not label:
                        break
                    words = sentence.strip().split(' ')
                    words = list(filter(lambda word:word!='',words))

                    tags = label.strip().split(' ')
                    tags = list(filter(lambda word:word!='',tags))

                    posTags = pos.strip().split(' ')
                    posTags = list(filter(lambda word:word!='',posTags))

                    sentence = f.readline()

                    if (len(words) != len(tags) or len(tags)!=len(posTags)):
                        print(input_file, ' words、labels、pos数不匹配：' + sentence + ' words length:' + str(
                            len(words)) + ' labels length:' + str(len(tags))+' pos length:'+str(len(posTags)))
                        continue
                    result.append(paddingAndEmbedding(input_file,words,tags,max_sequence_length,noEmbedding))
    return result

def input_fn(input_dir,shuffe,num_epochs,batch_size,max_sequence_length,one_sentence_words=None):
    global WV,POS_2_ID
    shapes = (([max_sequence_length,WV.vector_size],()),[max_sequence_length])
    types = ((tf.float32,tf.int32),tf.int32)
    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn,input_dir=input_dir,one_sentence_words=one_sentence_words,max_sequence_length = max_sequence_length),
        output_shapes=shapes,
        output_types=types
    )
    if shuffe:
        dataset = dataset.shuffle(buffer_size=10000).repeat(num_epochs)

    dataset = dataset.batch(batch_size)
    return dataset

def model_fn(features,labels,mode,params):
    is_training = (mode ==  tf.estimator.ModeKeys.TRAIN)
    #传入的features: ((句子每个单词向量，句子真实长度），句子每个tag索引).batchSize为
    features,lengths = features

    # LSTM
    print('构造LSTM层')
    #？

    t = tf.transpose(features, perm=[1, 0, 2])
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['hidden_units'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['hidden_units'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)

    print('LSTM联合层')
    output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=lengths)
    output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=lengths) #shape 49*batch_size*100
    output = tf.concat([output_fw, output_bw], axis=-1) #40*batch_size*200

    print('dropout')
    #?
    output = tf.transpose(output, perm=[1, 0, 2]) #batch_size*40*200
    output = tf.layers.dropout(output, rate=params['dropout_rate'], training=is_training)

    #全连接层
    logits = tf.layers.dense(output, len(TAG_2_ID)) #batch_size*40*len(tags)


    print('CRF层')
    # CRF

    crf_params = tf.get_variable("crf", [len(TAG_2_ID), len(TAG_2_ID)], dtype=tf.float32)
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
            indices = [item[1] for item in TAG_2_ID.items() if (item[0]!='<pad>'and item[0]!='O')]
            metrics = {
                'acc': tf.metrics.accuracy(labels, pred_ids, weights),
                # 'precision': tf.metrics.precision(labels,pred_ids,weights),
                # 'recall': tf.metrics.recall(labels,pred_ids,weights),
                # 'auc': tf.metrics.auc(labels,pred_ids,weights)
                'precision': precision(labels, pred_ids, len(TAG_2_ID), indices, weights),
                'recall': recall(labels, pred_ids, len(TAG_2_ID), indices, weights),
                'f1': f1(labels, pred_ids, len(TAG_2_ID), indices, weights),
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

#训练、评估、预测,sentece:要预测的句子
def main(FLAGS,sentence=None):

    tf.enable_eager_execution()
    # 配置哪块gpu可见
    # os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device_map

    # 在re train 的时候，才删除上一轮产出的文件，在predicted 的时候不做clean
    output_dir = os.path.join(FLAGS.root_dir,'output_'+FLAGS.sentence_mode)
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


    print('初始化标签-ID字典，33')
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

    print('构造estimator')
    estimator = tf.estimator.Estimator(model_fn,config=run_config,params=params)
    # estimator
    if FLAGS.ifTrain :
        print('获取训练数据。。。')
        train_inpf = functools.partial(input_fn, input_dir=(os.path.join(FLAGS.labeled_data_path, 'train')),
                                       shuffe=True, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size,max_sequence_length=FLAGS.max_sequence_length)
        train_total = len(list(train_inpf()))
        print('训练总数：'+str(train_total))
        num_train_steps = train_total/FLAGS.batch_size*FLAGS.num_epochs
        print('训练steps:'+str(num_train_steps))
        print('获取评估数据。。。')
        eval_inpf = functools.partial(input_fn, input_dir=(os.path.join(FLAGS.labeled_data_path, 'dev')),
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
        test_inpf = functools.partial(input_fn, input_dir=(os.path.join(FLAGS.labeled_data_path, 'test')),
                                      shuffe=False, num_epochs=1, batch_size=FLAGS.batch_size,max_sequence_length=FLAGS.max_sequence_length)

        predictions = estimator.predict(input_fn=test_inpf)
        pred_true = generator_fn(input_dir=(os.path.join(FLAGS.labeled_data_path, 'test')),max_sequence_length = FLAGS.max_sequence_length,noEmbedding=True)

        #取真实的tags
        targets = [x[1] for x in pred_true]
        print('预测总数：' + str(len(targets)))
        #预测结果
        pred = [x['pre_ids'] for x in list(predictions)]
        #预测分析
        report = flat_classification_report(y_pred=pred,y_true=targets)
        print(report)

        with open(os.path.join(output_dir,'predict_result.txt'),'w',encoding='utf8') as fw:
            fw.write(str(report))
            for target,predict in zip(pred_true,pred):
                (words,length),tags = target
                words = [words[i] for i in range(length)]
                labels = [ID_2_TAG[tags[i]] for i in range(length)]
                outputs = [ID_2_TAG[predict[i]] for i in range(length)]
                fw.write('原 文 ：'+' '.join(words))
                fw.write('\n')
                fw.write('人工标记： '+' '.join(labels))
                fw.write('\n')
                fw.write('预测结果： '+' '.join(outputs))
                fw.write('\n')
                fw.write('\n')

    if FLAGS.ifPredict and sentence:

        #初始化触发词集
        triggerDict = {}
        triggerDir = os.path.join(FLAGS.root_dir,'triggers')
        for triggerFile in os.listdir(triggerDir):
            with open(os.path.join(triggerDir,triggerFile),'r',encoding='utf8') as f:
                content = f.read()
            triggerDict[triggerFile.split('.')[0]] = set(content.split('\n'))

        #判断释放含触发词
        triggerContained = ''
        for oneKindTrigger in triggerDict.items():
            triggerType = oneKindTrigger[0]
            for word in oneKindTrigger[1]:
                if(sentence.find(word)!=-1):
                    triggerContained = triggerContained + (triggerType+':'+word)
                    break
            if(len(triggerType)==0):
                print('预测语句内无关注事实')
                return

        #分词、获取pos标签、去停用词
        words = SEGMENTOR.segment(sentence)

        # 去停用词
        newWords = []
        for word in words:
            if (word not in STOP_WORDS):
                newWords.append(word)

        pre_inf = functools.partial(input_fn, input_dir=None,one_sentence_words=newWords,
                                      shuffe=False, num_epochs=1, batch_size=FLAGS.batch_size,
                                      max_sequence_length=FLAGS.max_sequence_length)
        predictions = estimator.predict(input_fn=pre_inf)
        predictions = [x['pre_ids'] for x in list(predictions)][0]

        predictions = [ID_2_TAG[id] for id in predictions]


        print(sentence)
        print(' '.join(newWords))
        print(' '.join(predictions[0:len(newWords)]))
        release()
        return newWords,predictions
    release()

if __name__ == '__main__':
    # tf.enable_eager_execution()
    # rootdir = 'C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\'
    # initTagsAndWord2Vec(rootdir)
    # data = input_fn('C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\NERdata\\train', True, 1, 5, 50)
    # for v in data:
    #     print(v)
    pass