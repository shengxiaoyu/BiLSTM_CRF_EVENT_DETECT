#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

__doc__ = 'description'
__author__ = '13314409603@163.com'
from Config.config_parser import getParser
from Extract_Event.EventModel import EventFactory2
from extract_event import Event_Detection

def main():
    FLAGS = getParser()
    base_dir = os.path.join(os.path.join(os.path.join(FLAGS.root_dir,'labeled'),'Merge_for_second'),'test')
    true_events = []
    predict_examples = []
    for fileName in os.listdir(base_dir):
        with open(os.path.join(base_dir,fileName),'r',encoding='utf8') as f:
            #当前句子包含的所有事件
            events = []

            #当前句子
            currentSentence = f.readline()
            currentWords = currentSentence.strip().split()

            #第一层标签真实情况
            current_first_tags = f.readline().strip().split()

            #pos方式
            current_pos_tags = f.readline().strip().split()

            #每个事件的tag方式
            new_tags = f.readline().strip().split()
            events.append(EventFactory2(currentWords,new_tags))

            #下一个句子
            sentence = f.readline()
            while(sentence):
                if(sentence == currentSentence):
                    '''同一个句子'''
                    #去掉第一层标签行和pos行
                    f.readline()
                    f.readline()

                    #加入事件
                    new_tags = f.readline().strip().split()
                    events.append(EventFactory2(currentWords, new_tags))
                else:
                    '''不是同一个句子'''
                    #先合并tag，并加入训练集
                    predict_examples.append([currentWords,current_first_tags,current_pos_tags])

                    #将上个句子的事件抽取加入事件集
                    true_events.append(events)

                    #初始化
                    # 当前句子的实际事件
                    events = []
                    # 当前句子
                    currentSentence = sentence
                    currentWords = currentSentence.strip().split()

                    # 第一层标签真实情况
                    current_first_tags = f.readline().strip().split()

                    # pos方式
                    current_pos_tags = f.readline().strip().split()

                    # 每个事件的tag方式
                    new_tags = f.readline().strip().split()
                    events.append(EventFactory2(currentWords, new_tags))

                sentence = f.readline()

            #处理最后一个缓存
            predict_examples.append([currentWords, current_first_tags, current_pos_tags])
            true_events.append(events)

    extractor = Event_Detection(FLAGS)
    #基于第一层预测结果，使用模式匹配的方式构建事件
    events = extractor.extractor_from_words_posTags(predict_examples)

    fz = 0
    fm = 0
    true_events_total = 0
    pre_events_total = 0
    for events1,events2 in zip(true_events,events):
        true_events_total += len(true_events)
        pre_events_total += len(events)
        the_fz,the_fm = evalutaion(events1,events2)
        fz+=the_fz
        fm+=the_fm

    print('总共事件:'+true_events_total+'\t'+'含事件得分：'+str(fm))
    print('预测得到事件：'+pre_events_total+'\t'+'预测事件得分：'+str(fz))
    print('预测结果比:'+str(fz/fm))

    print('检查事件是否相同')
def merge(tagsList):
    mergedTags = ['O' for _ in range(len(tagsList[0]))]
    for tags in tagsList:
        for index,tag in enumerate(tags):
            if(tag.find('_Trigger')!=-1):
                mergedTags[index] = tag[:-8]
    return mergedTags

def evalutaion(events1,events2):
    fm = 0
    fz = 0
    if(events1!=None and events2==None):
        for event in events1:
            fm += event.get_score()
    else:
        for event in events1:
            if (event == None):
                continue

            #应得分
            fm += event.get_score()

            #实际得分
            for otherEvent in events2:
                if(event.type==otherEvent.type and event.trigger==otherEvent.trigger and event.tag_index_pair ==otherEvent.tag_index_pair):
                    the_fz= event.compare(otherEvent)
                    fz+=the_fz
                    found = True
                    break
    return fz,fm

if __name__ == '__main__':
    main()
    sys.exit(0)