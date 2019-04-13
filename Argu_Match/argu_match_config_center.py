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

def initNewTags(new_trigger_path,new_argu_path):
    global NEW_TRIGGER_TAGs,NEW_TRIGGER_LEN,NEW_TAG_2_ID,NEW_ID_2_TAG,NEW_ARGU_TAGs,NEW_ARGU_LEN,NEW_TAGs_LEN
    NEW_TAG_2_ID = {}
    NEW_ID_2_TAG = {}
    NEW_TRIGGER_TAGs = []
    NEW_TAG_2_ID['<pad>'] = 0
    NEW_ID_2_TAG[0] = '<pad>'
    index = 1
    with open(new_trigger_path,'r',encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            NEW_TAG_2_ID[line] = index
            NEW_ID_2_TAG[index] = line
            NEW_TRIGGER_TAGs.append(line)
            index += 1
    NEW_TRIGGER_LEN = len(NEW_TRIGGER_TAGs)

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

if __name__ == '__main__':
    pass