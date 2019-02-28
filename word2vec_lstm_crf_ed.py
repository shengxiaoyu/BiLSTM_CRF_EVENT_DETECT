#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'

import tensorflow as tf
import os
import functools

#device_map
#output_dir
#labeled_data_path

def mian(args):
    # 配置哪块gpu可见
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map

    # 在re train 的时候，才删除上一轮产出的文件，在predicted 的时候不做clean
    if args.clean and args.do_train:
        if os.path.exists(args.output_dir):
            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                    else:
                        os.remove(c_path)

            try:
                del_file(args.output_dir)
            except Exception as e:
                print(e)
                print('pleace remove the files of output dir and data.conf')
                exit(-1)
    # check output dir exists
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    session_config = tf.ConfigProto(
        # 是否打印使用设备的记录
        log_device_placement=False,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        # 是否允许自行使用合适的设备
        allow_soft_placement=True)

    run_config = tf.estimator.RunConfig(
        model_dir=args.output_dir,
        save_summary_steps=500,
        save_checkpoints_steps=500,
        session_config=session_config
    )

    if args.do_train and args.do_eval:
        train_file = os.path.join(args.output_dir,'train.tf_record')
        if not os.path.join(train_file):
            generator_fn(args.labeled_data_path,train_file)

def parse_fn(words,lables,max_squence_length):
    # Encode in Bytes for TF
    if(len(words)>max_squence_length):
        return words[0:max_squence_length],lables[0:max_squence_length]
    for i in range(len(words),max_squence_length):
        words.append('')

def generator_fn(input_dir,train_file,max_squence_length):
    writer = tf.python_io.TFRecordWriter()
    for root,dirs,files in os.walk(input_dir):
        features,targets = []
        for input_file in files:
            with open(os.path.join(input_dir,input_file),'r',encoding='utf8') as f:
                file_lines = f.readlines()
                i = 0
                while(i<len(file_lines)):
                    words = file_lines[i].strip().split(' ')
                    words = list(filter(lambda word:word!='',words))
                    if(i+1>=len(file_lines)):
                        break
                    labels = file_lines[i+1].strip().split(' ')
                    labels = list(filter(lambda word:word!='',labels))
                    if (len(words) != len(labels)):
                        print(input_file, ' words和labels数不匹配：' + file_lines[i] + ' words length:' + str(
                            len(words)) + ' labels length:' + str(len(labels)))
                        i += 2
                        continue
                    yield parse_fn(words,labels,max_squence_length)

def input_fn(input_dir):
    pass


def model_fn_builder():

    def model_fn(features,lables,mode,params):
        pass


if __name__ == '__main__':
    pass