#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__doc__ = 'description'
__author__ = '13314409603@163.com'

import functools
import os


import tensorflow as tf
from sklearn_crfsuite.metrics import flat_classification_report
import LSTM_CRF.config_center as CONFIG
import LSTM_CRF.input_fn as INPUT
import LSTM_CRF.model_fn as MODEL


#训练、评估、预测,sentece:要预测的句子
def main(FLAGS,sentences=None):
    print(FLAGS)

    tf.enable_eager_execution()
    # 配置哪块gpu可见
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device_map

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
        train_inpf = functools.partial(INPUT.input_fn, input_dir=(os.path.join(FLAGS.labeled_data_path, 'train')),
                                       shuffe=True, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size,max_sequence_length=FLAGS.max_sequence_length)
        train_total = len(list(train_inpf()))
        print('训练总数：'+str(train_total))
        num_train_steps = train_total/FLAGS.batch_size*FLAGS.num_epochs
        print('训练steps:'+str(num_train_steps))
        print('获取评估数据。。。')
        eval_inpf = functools.partial(INPUT.input_fn, input_dir=(os.path.join(FLAGS.labeled_data_path, 'dev')),
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
        test_inpf = functools.partial(INPUT.input_fn, input_dir=(os.path.join(FLAGS.labeled_data_path, 'test')),
                                      shuffe=False, num_epochs=1, batch_size=FLAGS.batch_size,max_sequence_length=FLAGS.max_sequence_length)

        predictions = estimator.predict(input_fn=test_inpf)
        pred_true = INPUT.generator_fn(input_dir=(os.path.join(FLAGS.labeled_data_path, 'test')),max_sequence_length = FLAGS.max_sequence_length,noEmbedding=True)

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
                (words,length,postags),tags = target
                words = [words[i] for i in range(length)]
                labels = [CONFIG.ID_2_TAG[tags[i]] for i in range(length)]
                outputs = [CONFIG.ID_2_TAG[predict[i]] for i in range(length)]
                fw.write('原 文 ：'+' '.join(words))
                fw.write('\n')
                fw.write('人工标记： '+' '.join(labels))
                fw.write('\n')
                fw.write('预测结果： '+' '.join(outputs))
                fw.write('\n')
                fw.write('\n')

    if FLAGS.ifPredict and sentences:
        sentences_words_posTags = []
        for sentence in sentences:
            #分词、获取pos标签、去停用词
            words = CONFIG.SEGMENTOR.segment(sentence)
            postags = CONFIG.POSTAGGER.postag(words)

            # 去停用词
            newWords = []
            newPosTags = []
            for word, pos in zip(words, postags):
                if (word not in CONFIG.STOP_WORDS):
                    newWords.append(word)
                    newPosTags.append(pos)
            sentences_words_posTags.append([newWords,newPosTags])
        pre_inf = functools.partial(INPUT.input_fn, input_dir=None,sentences_words_posTags=sentences_words_posTags,
                                      shuffe=False, num_epochs=1, batch_size=FLAGS.batch_size,
                                      max_sequence_length=FLAGS.max_sequence_length)
        predictions = estimator.predict(input_fn=pre_inf)
        predictions = [x['pre_ids'] for x in list(predictions)]

        result = []
        for one_sentence_words_posTags,pre_ids in zip(sentences_words_posTags,predictions):
            words = one_sentence_words_posTags[0]
            pre_tags = [CONFIG.ID_2_TAG[id]for id in pre_ids]
            result.append([words,pre_tags])
            print(' '.join(words))
            print(' '.join(pre_tags[0:len(words)]))
        CONFIG.release()
        return result
    CONFIG.release()

if __name__ == '__main__':
    # tf.enable_eager_execution()
    # rootdir = 'C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\'
    # initTagsAndWord2Vec(rootdir)
    # data = input_fn('C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\NERdata\\train', True, 1, 5, 50)
    # for v in data:
    #     print(v)
    pass