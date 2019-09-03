#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'
from conflict_detect.event_alignment.alignmenter import alignmenter
from conflict_detect.data_pre_process.handle_single_file import single_file_handler
import os
from check_conflict.check_conflict import checker_conflict
from model import Relation

def run():
    ali = alignmenter()
    checker = checker_conflict()
    file_handler = single_file_handler()
    root = r'A:\Bi-LSTM+CRF\brat\relabled'

    result = {}

    #冲突事件集合
    contradictory = {}
    #冲突事件被预测为冲突
    contradictory['p-Contradictory'] = 0
    #蕴含
    contradictory['p-Entailment'] = 0
    #没有检测出来
    contradictory['p-non'] = 0
    #放入result集合
    result['Contradictory'] = contradictory

    entailment = {}
    entailment['p-Contradictory'] = 0
    entailment['p-Entailment'] = 0
    entailment['p-non'] = 0
    result['Entailment'] = entailment

    #标记数据中没有的事件被预测为冲突或者蕴含
    non = {}
    non['p-Contradictory'] = 0
    non['p-Entailment'] = 0
    result['non'] = non

    result_events = {}

    # 冲突事件集合
    contradictory_events = {}
    # 冲突事件被预测为冲突
    contradictory_events['p-Contradictory'] = []
    # 蕴含
    contradictory_events['p-Entailment'] = []
    # 没有检测出来
    contradictory_events['p-non'] = []
    # 放入result集合
    result_events['Contradictory'] = contradictory_events

    entailment_events = {}
    entailment_events['p-Contradictory'] = []
    entailment_events['p-Entailment'] = []
    entailment_events['p-non'] = []
    result_events['Entailment'] = entailment_events

    # 标记数据中没有的事件被预测为冲突或者蕴含
    non_events = {}
    non_events['p-Contradictory'] = []
    non_events['p-Entailment'] = []
    result_events['non'] = non_events


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
                event_map,event_relations = file_handler.handler(os.path.join(index_dir,file),origin_file)

                events = list(event_map.values())
                my_event_relations = []

                for i in range(len(events)):
                    for j in range(i+1,len(events)):
                        if (ali.alignment(events[i], events[j])):
                            if (checker.check(events[i], events[j])):
                                my_event_relations.append(Relation(None,'Contradictory',events[i],events[j]))
                            else:
                                my_event_relations.append(Relation(None,'Entailment',events[i],events[j]))
                cal(result,event_relations,my_event_relations,result_events)
    save_root = r'A:\Bi-LSTM+CRF\bert\data'
    with open(os.path.join(save_root, 'event_relations.txt'), 'w', encoding='utf8') as writer:
        writer.write('\t\tp-Contradictory\tp-Entailment\tp-Non\n')
        for (key,val) in result.items():
            writer.write(key+'\t'+str(val['p-Contradictory'])+'\t'+str(val['p-Entailment'])+'\t')
            if(key!='non'):
                writer.write('\t'+str(val['p-non'])+'\n')
            else:
                writer.write('\n')


def cal(result,relations,pre_relations,result_events):
    if(relations==None or len(relations)==0):
        if(pre_relations!=None and len(pre_relations)>0):
            for relation in pre_relations:
                result['non']['p-'+relation.type] += 1
                result_events['non']['p-'+relation.type].append(relation)
            return
    if(pre_relations==None or len(pre_relations)==0):
        if(relations!=None and len(relations)>0):
            for relation in relations:
                result[relation.type]['p-non'] += 1
                result_events[relation.type]['p-non'].append(relation)
            return

    for relation in relations:
        found = False
        for pre_relation in pre_relations:
            if(relation.__eq__(pre_relation)):
                result[relation.type]['p-'+pre_relation.type] += 1
                result_events[relation.type]['p-'+pre_relation.type].append(relation)
                found = True
                break
        if(not found):
            result[relation.type]['p-non'] += 1
            result_events[relation.type]['p-non'].append(relation)

    for pre_relation in pre_relations:
        found = False
        for relation in relations:
            if(pre_relation.__eq__(relation)):
                found = True
                break
        if(not found):
            result['non']['p-'+pre_relation.type] += 1
            result_events['non']['p-'+pre_relation.type].append(pre_relation)

if __name__ == '__main__':
    run()