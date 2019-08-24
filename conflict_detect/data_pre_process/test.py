#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from model import event_types

__doc__ = 'description'
__author__ = '13314409603@163.com'

import sys
from conflict_detect.data_pre_process.handle_single_file import single_file_handler
from conflict_detect.edit_distance_similar.similar_calculator import EditDistanceSimilarCalculator
import os

cal = EditDistanceSimilarCalculator()

def getSeparation():
    handler = single_file_handler()
    root = r'A:\Bi-LSTM+CRF\brat\labeled'
    save_root = r'A:\Bi-LSTM+CRF\event_conflict'
    writer = open(os.path.join(save_root,'separation.txt'),'w',encoding='utf8')
    for dir_name in os.listdir(root):
        dir = os.path.join(root, dir_name)
        if (not os.path.isdir(dir)):
            continue
        for index_name in os.listdir(dir):
            index_dir = os.path.join(dir, index_name)
            if (not os.path.isdir(index_dir)):
                continue
            for file in os.listdir(index_dir):
                '''遍历每个文件获取events'''
                if (file.find('.ann') == -1):
                    continue

                origin_file = os.path.join(index_dir, file.replace('ann', 'txt'))
                if (not os.path.exists(origin_file)):
                    continue
                event_map, _ = handler.handler(os.path.join(index_dir, file), origin_file)
                sentences = []
                for event in event_map.values():
                    if (event.type == event_types.SEPARATION):
                        sentences.append('\t'+event.sentence+"\n")
                if(len(sentences)>1):
                    writer.write(origin_file+'\n')
                    for sentence in sentences:
                        writer.write(sentence)
def getDerailedTrigger():
    handler = single_file_handler()
    root = r'A:\Bi-LSTM+CRF\brat\relabled'
    derailed_trigger_set = set()
    for dir_name in os.listdir(root):
        dir = os.path.join(root, dir_name)
        if (not os.path.isdir(dir)):
            continue
        for index_name in os.listdir(dir):
            index_dir = os.path.join(dir, index_name)
            if (not os.path.isdir(index_dir)):
                continue
            for file in os.listdir(index_dir):
                '''遍历每个文件获取events'''
                if (file.find('.ann') == -1):
                    continue

                origin_file = os.path.join(index_dir, file.replace('ann', 'txt'))
                if (not os.path.exists(origin_file)):
                    continue

                event_map, _ = handler.handler(os.path.join(index_dir, file), origin_file)
                for event in event_map.values():
                    if (event.type == event_types.DERAILED):
                        derailed_trigger_set.add(event.trigger.value)
        save_root = r'A:\Bi-LSTM+CRF\event_conflict'
        with open(os.path.join(save_root, 'derailed_trigger.txt'), 'a', encoding='utf8') as writer:
            writer.write('\n'.join(derailed_trigger_set))

def getWealtherPerson():
    handler = single_file_handler()
    root = r'A:\Bi-LSTM+CRF\brat\relabled'
    famliy_conflict_set = set()
    for dir_name in os.listdir(root):
        dir = os.path.join(root, dir_name)
        if (not os.path.isdir(dir)):
            continue
        for index_name in os.listdir(dir):
            index_dir = os.path.join(dir, index_name)
            if (not os.path.isdir(index_dir)):
                continue
            for file in os.listdir(index_dir):
                '''遍历每个文件获取events'''
                if (file.find('.ann') == -1):
                    continue

                origin_file = os.path.join(index_dir, file.replace('ann', 'txt'))
                if (not os.path.exists(origin_file)):
                    continue

                event_map, _ = handler.handler(os.path.join(index_dir, file), origin_file)
                for event in event_map.values():
                    if (event.type == event_types.WEALTH and hasattr(event, 'Whose')):
                        print(event.Whose.value)
                        if(event.Whose.value=='双方' or event.Whose.value=='每人'):
                            print(event.sentence)

def whetherWealthHasNegation():
    handler = single_file_handler()
    root = r'A:\Bi-LSTM+CRF\brat\relabled'
    famliy_conflict_set = set()
    for dir_name in os.listdir(root):
        dir = os.path.join(root, dir_name)
        if (not os.path.isdir(dir)):
            continue
        for index_name in os.listdir(dir):
            index_dir = os.path.join(dir, index_name)
            if (not os.path.isdir(index_dir)):
                continue
            for file in os.listdir(index_dir):
                '''遍历每个文件获取events'''
                if (file.find('.ann') == -1):
                    continue

                origin_file = os.path.join(index_dir, file.replace('ann', 'txt'))
                if (not os.path.exists(origin_file)):
                    continue

                event_map, _ = handler.handler(os.path.join(index_dir, file), origin_file)
                for event in event_map.values():
                    if (event.type == event_types.WEALTH and hasattr(event,'Negation')):
                        print(event.sentence)

def getFamliyConflictTrigger():
    handler = single_file_handler()
    root = r'A:\Bi-LSTM+CRF\brat\relabled'
    famliy_conflict_set = set()
    for dir_name in os.listdir(root):
        dir = os.path.join(root, dir_name)
        if (not os.path.isdir(dir)):
            continue
        for index_name in os.listdir(dir):
            index_dir = os.path.join(dir, index_name)
            if (not os.path.isdir(index_dir)):
                continue
            for file in os.listdir(index_dir):
                '''遍历每个文件获取events'''
                if (file.find('.ann') == -1):
                    continue

                origin_file = os.path.join(index_dir, file.replace('ann', 'txt'))
                if (not os.path.exists(origin_file)):
                    continue

                event_map, _ = handler.handler(os.path.join(index_dir, file), origin_file)
                for event in event_map.values():
                    if (event.type == event_types.FAMILY_CONFLICT):
                        famliy_conflict_set.add(event.trigger.value)
    save_root = r'A:\Bi-LSTM+CRF\event_conflict'
    with open(os.path.join(save_root, 'family_conflict_trigger.txt'), 'a', encoding='utf8') as writer:
        writer.write('\n'.join(famliy_conflict_set))

def getRemarryTrigger():
    handler = single_file_handler()
    root = r'A:\Bi-LSTM+CRF\brat\relabled'
    remarry_set = set()
    for dir_name in os.listdir(root):
        dir = os.path.join(root, dir_name)
        if (not os.path.isdir(dir)):
            continue
        for index_name in os.listdir(dir):
            index_dir = os.path.join(dir, index_name)
            if (not os.path.isdir(index_dir)):
                continue
            for file in os.listdir(index_dir):
                '''遍历每个文件获取events'''
                if (file.find('.ann') == -1):
                    continue

                origin_file = os.path.join(index_dir, file.replace('ann', 'txt'))
                if (not os.path.exists(origin_file)):
                    continue

                event_map, _ = handler.handler(os.path.join(index_dir, file), origin_file)
                for event in event_map.values():
                    if(event.type==event_types.REMARRY):
                        remarry_set.add(event.trigger.value)
    save_root = r'A:\Bi-LSTM+CRF\event_conflict'
    with open(os.path.join(save_root, 'remarry_trigger.txt'), 'a', encoding='utf8') as writer:
        writer.write('\n'.join(remarry_set))

def getNegatedPhrase():
    handler = single_file_handler()
    root = r'A:\Bi-LSTM+CRF\brat\labeled'
    negated_set = set()
    for dir_name in os.listdir(root):
        dir = os.path.join(root, dir_name)
        if (not os.path.isdir(dir)):
            continue
        for index_name in os.listdir(dir):
            index_dir = os.path.join(dir, index_name)
            if (not os.path.isdir(index_dir)):
                continue
            for file in os.listdir(index_dir):
                '''遍历每个文件获取events'''
                if (file.find('.ann') == -1):
                    continue

                origin_file = os.path.join(index_dir, file.replace('ann', 'txt'))
                if (not os.path.exists(origin_file)):
                    continue

                event_map, _ = handler.handler(os.path.join(index_dir, file), origin_file)
                for event in event_map.values():
                    if(hasattr(event,'Negation')):
                        if(event.Negation.value=='不是事' or event.Negation.value=='怀疑' or event.Negation.value=='亦承认'):
                            print('found')
                        negated_set.add(event.Negation.value)
    save_root = r'A:\Bi-LSTM+CRF\event_conflict'
    with open(os.path.join(save_root, 'negated.txt'), 'a', encoding='utf8') as writer:
        writer.write('\n'.join(negated_set))

#
def getTargetPhrase():
    handler = single_file_handler()
    root = r'A:\Bi-LSTM+CRF\brat\labeled'
    types_of_conflic = set()
    types_of_habits = set()
    types_of_wealth = set()
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

                event_map,_ = handler.handler(os.path.join(index_dir,file),origin_file)

                '''获取所有需要判断是否相似的短语并存储'''
                family_confilct_trigger = []
                '''孩子姓名'''
                children_name = []
                '''财产'''
                wealth_name = []
                '''时间'''
                time = []
                '''person'''
                person = []
                '''habit'''
                habit_name = []

                #存储事件
                family_conflict = []
                wealth = []

                '''遍历事件'''
                for event in event_map.values():
                    if(event.type==event_types.FAMILY_CONFLICT):
                        family_confilct_trigger.append(event.trigger.value)
                        types_of_conflic.add(event.trigger.value)
                        family_conflict.append(event)
                    elif(event.type==event_types.BE_BORN):
                        if(hasattr(event,'ChildName')):
                            children_name.append(event.ChildName.value)
                    elif(event.type==event_types.WEALTH):
                        wealth_name.append(event.trigger.value)
                        types_of_wealth.add(event.trigger.value)
                        wealth.append(event)
                    elif(event.type==event_types.MARRY):
                        if(hasattr(event,'Time')):
                            time.append(event.Time.value)
                    elif(event.type==event_types.DIVORCE_LAWSUIT):
                        if(hasattr(event,'Perpetrators')):
                            person.append(event.Perpetrators.value)
                        if(hasattr(event,'Victim')):
                            person.append(event.Victim.value)
                        if(hasattr(event,'Time')):
                            time.append(event.Time.value)
                    elif(event.type==event_types.BAD_HABIT):
                        if(hasattr(event,'Participant')):
                            person.append(event.Participant.value)
                        habit_name.append(event.trigger.value)
                        types_of_habits.add(event.trigger.value)
                    elif(event.type==event_types.DERAILED):
                        if(hasattr(event,'Derailer')):
                            person.append(event.Derailer.value)
                        if(hasattr(event,'Time')):
                            time.append(event.Time.value)
                    elif(event.type==event_types.SEPARATION):
                        if(hasattr(event,'BeginTime')):
                            time.append(event.BeginTime.value)
                        if(hasattr(event,'EndTime')):
                            time.append(event.EndTime.value)
                    elif(event.type==event_types.DIVORCE_LAWSUIT):
                        if(hasattr(event,'SueTime')):
                            time.append(event.SueTime.value)
                        if(hasattr(event,'Initiator')):
                            person.append(event.Initiator.value)
                    elif(event.type==event_types.DEBT):
                        if(hasattr(event,'Creditor')):
                            person.append(event.Creditor.value)
                    elif (event.type == event_types.CREDIT):
                        if (hasattr(event, 'Debtor')):
                            person.append(event.Debtor.value)
                save_root = r'A:\Bi-LSTM+CRF\bert\data'
                with open(os.path.join(save_root,'train1.txt'),'a',encoding='utf8') as writer:
                    write_events(writer,family_conflict)
                    write_events(writer,wealth)


def write_append(writer,data):
    if(len(data)>1):
        for i in range(len(data)):
            for j in range(i+1,len(data)):
                # writer.write(data[i]+'\n')
                writer.write(data[i]+'\t'+data[j]+'\t'+str(cal.editDistance(data[i],data[j]))+'\t'+str(cal.cal(data[i],data[j]))+'\n')
        writer.write('\n')
def write_events(writer,events):
    if(len(events)>1):
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                writer.write(events[i].trigger.value+'\t'+events[j].trigger.value+'\t'+str(cal.editDistance(events[i].trigger.value,events[j].trigger.value))+'\t'+str(cal.cal(events[i].trigger.value,events[j].trigger.value))+'\n')
                writer.write(events[i].sentence+'\t'+events[j].sentence+'\n')
        writer.write('\n')

if __name__ == '__main__':
    getSeparation()
    sys.exit(0)