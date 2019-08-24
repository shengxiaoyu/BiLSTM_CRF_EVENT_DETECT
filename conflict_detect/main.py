#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__doc__ = 'description'
__author__ = '13314409603@163.com'
from conflict_detect.event_alignment.alignmenter import alignmenter
from conflict_detect.data_pre_process.handle_single_file import single_file_handler
from conflict_detect.check_conflict.check_conflict import checker_conflict
import os

def run():
    ali = alignmenter()
    checker = checker_conflict()
    file_handler = single_file_handler()
    root = r'A:\Bi-LSTM+CRF\brat\labeled'

    save_root = r'A:\Bi-LSTM+CRF\bert\data'
    writer = open(os.path.join(save_root,'event_relations.txt'),'w',encoding='utf8')

    for dir_name in os.listdir(root):
        dir = os.path.join(root,dir_name)
        if(not os.path.isdir(dir)):
            continue
        for index_name in os.listdir(dir):
            index_dir = os.path.join(dir,index_name)
            if(not os.path.isdir(index_dir)):
                continue
            for file in os.listdir(index_dir):
                '''遍历每个文件获取events'''
                if(file.find('.ann')==-1):
                    continue
                origin_file = os.path.join(index_dir,file.replace('ann','txt'))
                if(not os.path.exists(origin_file)):
                    continue
                if(origin_file.find('1164467')==-1):
                    continue
                event_map,event_relations = file_handler.handler(os.path.join(index_dir,file),origin_file)
                writer.write(origin_file+":\n")
                writer.write('\t标注数据：\n')
                for event_relation in event_relations:
                    writer.write("\t\t"+str(event_relation.type)+":\t"+str(event_relation.arg1)+"\t"+str(event_relation.arg2)+"\n")
                writer.write('\t自动识别：\n')
                events = list(event_map.values())
                for i in range(len(events)):
                    for j in range(i+1,len(events)):
                        if (ali.alignment(events[i], events[j])):
                            if (checker.check(events[i], events[j])):
                                writer.write("\t\tContradictor" + ":\t" + str(events[i]) + "\t" + str(events[j]) + '\n')
                            else:
                                writer.write("\t\tEntailment" + ":\t" + str(events[i]) + "\t" + str(events[j]) + '\n')
    writer.close()


if __name__ == '__main__':
    run()