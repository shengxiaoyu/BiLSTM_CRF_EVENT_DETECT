#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

__doc__ = 'description'
__author__ = '13314409603@163.com'
from Config.config_parser import getParser
from Extract_Event.EventModel import EventFactory2
from Extract_Event.extract_event import Event_Detection

def main():
    FLAGS = getParser()
    base_dir = os.path.join(os.path.join(os.path.join(FLAGS.root_dir,'labeled'),'Merge_for_second'),'test')
    true_events = []
    predict_examples = []
    for fileName in os.listdir(base_dir):
        with open(os.path.join(base_dir,fileName),'r',encoding='utf8') as f:
            currentSentence = f.readline()
            while(currentSentence):
                # 当前句子
                currentWords = currentSentence.strip().split()

                # 第一层标签真实情况
                f.readline()

                # pos方式
                current_pos_tags = f.readline().strip().split()

                # 每个事件的tag方式
                current_second_tags = f.readline().strip().split()
                true_events.append(EventFactory2(currentWords, current_second_tags))

                predict_examples.append([currentWords, current_second_tags, current_pos_tags])
                # 下一个句子
                currentSentence = f.readline()

    extractor = Event_Detection(FLAGS,output_path='output_1_5_fullPos_trigger_Spe')
    #单句单事实直接准确匹配
    events = extractor.extractor_from_words_posTags(predict_examples)

    fz = 0
    fm = 0
    true_events_total = len(true_events)
    pre_events_total = len(events)
    for event1, event2 in zip(true_events,events):

        the_fz, the_fm = evalutaion(event1,event2)
        fz += the_fz
        fm += the_fm

    print('总共事件:'+str(true_events_total)+'\t'+'含事件得分：'+str(fm))
    print('预测得到事件：'+str(pre_events_total)+'\t'+'预测事件得分：'+str(fz))
    print('预测结果比:'+str(fz/fm))

    print('检查事件是否相同')
def merge(tagsList):
    mergedTags = ['O' for _ in range(len(tagsList[0]))]
    for tags in tagsList:
        for index,tag in enumerate(tags):
            if(tag.find('_Trigger')!=-1):
                mergedTags[index] = tag[:-8]
    return mergedTags

def evalutaion(event1,event2):
    if(event1==None ):
        return 0,0
    fm = event1.get_score()
    if (event2) :
        if(event1.type != event2.type):
            fz = 0
        else:
            fz = event1.compare(event2)
    else:
        fz = 0
    return fz, fm

if __name__ == '__main__':
    main()
    sys.exit(0)