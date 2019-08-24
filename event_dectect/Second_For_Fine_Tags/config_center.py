#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'


# NEW_TAGs = None
NEW_TAG_2_ID = None
NEW_ID_2_TAG = None
NEW_TAGs_LEN = 0
NEW_TRIGGER_TAGs = None
NEW_TRIGGER_LEN = 0
NEW_ARGU_TAGs = None
NEW_ARGU_LEN = 0

NEW_TRIGGER_TYPE_ONEHOT_DICT = {}

ifInited =False

def initNewTags(new_trigger_path,new_argu_path):

    global ifInited,NEW_TRIGGER_TAGs,NEW_TRIGGER_LEN,NEW_TAG_2_ID,NEW_ID_2_TAG,NEW_ARGU_TAGs,NEW_ARGU_LEN,NEW_TAGs_LEN,NEW_TRIGGER_TYPE_ONEHOT_DICT
    if(not ifInited):
        NEW_TAG_2_ID = {}
        NEW_ID_2_TAG = {}
        NEW_TRIGGER_TAGs = []
        NEW_TAG_2_ID['<pad>'] = 0
        NEW_ID_2_TAG[0] = '<pad>'

        NEW_TAG_2_ID['O'] = 1
        NEW_ID_2_TAG[1] = 'O'

        #事实类别：
        event_set = set()

        index = 2
        with open(new_trigger_path,'r',encoding='utf8') as f:
            for line in f.readlines():
                line = line.strip()
                NEW_TAG_2_ID[line] = index
                NEW_ID_2_TAG[index] = line
                NEW_TRIGGER_TAGs.append(line)
                index += 1
                event_set.add(line[2:-8])
        NEW_TRIGGER_LEN = len(NEW_TRIGGER_TAGs)

        #根据事实类别set构造ONE_HOT字典
        for type in event_set:
            NEW_TRIGGER_TYPE_ONEHOT_DICT[type] = [1 if type==the_type else 0 for the_type in event_set]

        NEW_ARGU_TAGs = []
        with open(new_argu_path,'r',encoding='utf8') as f:
            for line in f.readlines():
                line = line.strip()
                NEW_ARGU_TAGs.append(line)
                NEW_TAG_2_ID[line] = index
                NEW_ID_2_TAG[index] = line
                index += 1
        NEW_ARGU_LEN = len(NEW_ARGU_TAGs)

        NEW_TAGs_LEN = len(NEW_TAG_2_ID)

        ifInited = True

if __name__ == '__main__':
    pass