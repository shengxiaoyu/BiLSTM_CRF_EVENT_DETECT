#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'
import os
from Config.config_parser import getParser
import LSTM_CRF.word2vec_lstm_crf_ed as run
import EventModel
root_dir = 'C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF'




#判断是否含有关注事实触发词
def ifContainTrigger(sentence):
    # 初始化触发词集
    triggerDict = {}
    triggerDir = os.path.join(root_dir, 'triggers')
    for triggerFile in os.listdir(triggerDir):
        with open(os.path.join(triggerDir, triggerFile), 'r', encoding='utf8') as f:
            content = f.read()
        triggerDict[triggerFile.split('.')[0]] = set(content.split('\n'))

    # 判断释放含触发词
    triggerContained = ''
    for oneKindTrigger in triggerDict.items():
        triggerType = oneKindTrigger[0]
        for word in oneKindTrigger[1]:
            if (sentence.find(word) != -1):
                triggerContained = triggerContained + (triggerType + ':' + word)
                break
        if (len(triggerType) == 0):
            return False
    return True

def extractor(sentence):
    hasTrigger = ifContainTrigger(sentence)
    if(not hasTrigger):
        print('该句子中无关注事实：'+sentence)

    #调用预测接口
    FLAGS = getParser()
    FLAGS.ifTrain = False
    FLAGS.ifTest = False
    FLAGS.ifPredict = True
    words, tags = run.main(FLAGS, sentence)
    print(' '.join(words))
    print(' '.join(tags))
    # words = '原 被告 2007年 11月 网上 相识 恋爱 2008年 3月 17日 登记 结婚 ×××× 年×× 月×× 日 生育 女儿 戴 乙 2012年 6月 1日 生育 女儿 罗某乙 <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>'.split()
    # tags = 'O O B_Time I_Time O B_Know B_BeInLove B_Time I_Time I_Time B_Marry I_Marry B_Time I_Time I_Time I_Time B_Bear B_Gender B_Name I_Name B_Time I_Time I_Time B_Bear B_Gender B_Name <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>'.split()

    events = []

    triggers = []
    # 获取触发词tag
    with open(os.path.join(FLAGS.root_dir, 'triggerLabels.txt'), 'r', encoding='utf8') as f:
        for line in f.readlines():
            triggers.append(line.strip())

    def saveOneEvent(trigger,beginIndex,endIndex):
        completeWord = ''
        for word in words[beginIndex:endIndex+1]:
            completeWord += word
        # 先把前面的事件确定了
        event = EventModel.EventFactory(trigger,completeWord, beginIndex, endIndex)
        event.fitArgument(words, tags)
        events.append(event)

    hasBegin = False
    currentTraigger = None
    beginIndex = 0
    endIndex = 0

    for index,word in enumerate(tags):
        if(word in triggers): #如果是触发词
            if(word.find('B_')!=-1): #如果是B_开头
                if(hasBegin):#如果前面有触发词还在统计
                    saveOneEvent(currentTraigger,beginIndex,endIndex)
                #新起一个事件
                hasBegin=True
                currentTraigger = word[2:]
                beginIndex = index
                endIndex = index
            else:#I_开头
                if(hasBegin): #此时正在一个触发词的查找范围内
                    if(word.find(currentTraigger)!=-1): #同一个触发词
                        endIndex = index
                    else:#此时在找触发词，但是来了个其他触发词的I_
                        saveOneEvent(currentTraigger, beginIndex, endIndex)
                        hasBegin = True
                        currentTraigger = word[2:]
                        beginIndex = index
                        endIndex = index
                else:#此时没有找触发词直接来了个I_
                    hasBegin = True
                    currentTraigger = word[2:]
                    beginIndex = index
                    endIndex = index
        else:
            if(hasBegin):#查找触发词正常结束
                saveOneEvent(currentTraigger, beginIndex, endIndex)
                hasBegin = False
                beginIndex = 0
                endIndex = 0
    return events

if __name__ == '__main__':
    # events = extractor('被告季某辩称，原告所陈述的事实理由不正确，原被告于2009年农历正月认识，××××年××月××日生育一女，婚后为了抚养小孩发生了争吵，被告也曾不小心碰伤了原告。')
    # events2 = extractor('原、被告于2007年11月于网上相识恋爱，200 8年3月17日登记结婚，××××年××月××日生育女儿戴某乙，2012 年6月1日生育女儿罗某乙。')
    # events3 = extractor('原、被告系夫妻关系，1981年自由恋爱。')
    # events4 = extractor('81年12月份结婚登记82年10月份结婚，婚后育一子李冉，现已结婚，独立生活。')
    events5 = extractor('曾多次向人民法院提起离婚诉讼，被判决不准离婚，')

    print('end')
    pass