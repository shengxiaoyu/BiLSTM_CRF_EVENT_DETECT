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
    base_dir = os.path.join(os.path.join(os.path.join(FLAGS.root_dir,'labeled'),'Spe'),'test')
    true_events = []
    predict_examples = []
    for fileName in os.listdir(base_dir):
        with open(os.path.join(base_dir,fileName),'r',encoding='utf8') as f:
            #当前句子的实际事件
            events = []
            #当前句子
            currentSentence = f.readline()
            currentWords = currentSentence.strip().split()
            #每个事件的tag方式
            new_tags = f.readline().strip().split()
            events.append(EventFactory2(currentWords,new_tags))
            #当前句子对应的所有tag方式
            new_tags_list = []
            new_tags_list.append(new_tags)
            #当前句子的pos标签
            current_pos_tags = f.readline().strip().split()

            #下一个句子
            sentence = f.readline()
            while(sentence):
                if(sentence == currentSentence):
                    '''同一个句子'''
                    new_tags = f.readline().strip().split()
                    events.append(EventFactory2(currentWords, new_tags))
                    new_tags_list.append(new_tags)
                    #去掉pos行
                    f.readline()
                else:
                    '''不是同一个句子'''
                    #先合并tag，并加入训练集
                    old_merge_tags = merge(new_tags_list)
                    predict_examples.append([currentWords,old_merge_tags,current_pos_tags])

                    #将上个句子的事件抽取加入事件集
                    true_events.append(events)

                    #初始化
                    # 当前句子的实际事件
                    events = []
                    # 当前句子
                    currentSentence = sentence
                    currentWords = currentSentence.strip().split()
                    # 每个事件的tag方式
                    new_tags = f.readline().strip().split()
                    events.append(EventFactory2(currentWords, new_tags))
                    # 当前句子对应的所有tag方式
                    new_tags_list = []
                    new_tags_list.append(new_tags)
                    # 当前句子的pos标签
                    current_pos_tags = f.readline().strip().split()
                sentence = f.readline()

            #处理最后一个缓存
            old_merge_tags = merge(new_tags_list)
            predict_examples.append([currentWords, old_merge_tags, current_pos_tags])
            true_events.append(events)

    extractor = Event_Detection(FLAGS)
    events = extractor.extractor_from_words_posTags(predict_examples)

    fz = 0
    fm = 0
    for events1,events2 in zip(true_events,events):
        the_fz,the_fm = evalutaion(events1,events2)
        fz+=the_fz
        fm+=the_fm
    print(fz)
    print(fm)
    print(fz/fm)
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
            if(event==None):
                continue
            found = False
            for otherEvent in events2:
                if(event.type==otherEvent.type and event.trigger==otherEvent.trigger and event.tag_index_pair ==otherEvent.tag_index_pair):
                    the_fz= event.compare(otherEvent)
                    the_fm = event.get_score()
                    fz+=the_fz
                    fm+=the_fm
                    found = True
                    break
            if(not found):
                fm += event.get_score()
    return fz,fm

if __name__ == '__main__':
    main()
    sys.exit(0)