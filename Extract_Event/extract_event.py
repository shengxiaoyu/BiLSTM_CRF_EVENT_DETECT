#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'
from pyltp import SentenceSplitter

import LSTM_CRF.config_center as CONFIG
import LSTM_CRF.word2vec_lstm_crf_ed as run
from Config.config_parser import getParser
import Extract_Event.EventModel as EventModel

root_dir = 'C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF'

#判断是否含有关注事实触发词
def ifContainTrigger(sentence):
    # 初始化触发词集
    triggerDict = CONFIG.TRIGGER_WORDS_DICT
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

def extractor(paragraph):
    CONFIG.init(root_dir)
    sentenceSplitter = SentenceSplitter()
    sentences = []
    for sentence in sentenceSplitter.split(paragraph):
        hasTrigger = ifContainTrigger(sentence)
        if(not hasTrigger):
            print('该句子中无关注事实：'+sentence)
        else:
            sentences.append(sentence)
    #调用预测接口
    FLAGS = getParser()
    FLAGS.ifTrain = False
    FLAGS.ifTest = False
    FLAGS.ifPredict = True
    predictions = run.main(FLAGS, sentences)
    events = []
    # 获取触发词tag
    triggers = CONFIG.TRIGGER_TAGs


    def saveOneEvent(trigger,beginIndex,endIndex,words,tags):
        completeWord = ''
        for word in words[beginIndex:endIndex+1]:
            completeWord += word
        # 先把前面的事件确定了
        event = EventModel.EventFactory(trigger, completeWord, beginIndex, endIndex)
        event.fitArgument(words, tags)
        events.append(event)

    for words,tags in predictions:

        hasBegin = False
        currentTraigger = None
        beginIndex = 0
        endIndex = 0

        for index,word in enumerate(tags):
            if(word in triggers): #如果是触发词
                if(word.find('B_')!=-1): #如果是B_开头
                    if(hasBegin):#如果前面有触发词还在统计
                        saveOneEvent(currentTraigger,beginIndex,endIndex,words,tags)
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
                            saveOneEvent(currentTraigger, beginIndex, endIndex,words,tags)
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
                    saveOneEvent(currentTraigger, beginIndex, endIndex,words,tags)
                    hasBegin = False
                    beginIndex = 0
                    endIndex = 0
    return events

if __name__ == '__main__':
    events = extractor('原、被告系夫妻关系，1981年自由恋爱。81年12月份结婚登记82年10月份结婚，婚后育一子李冉，现已结婚，独立生活。'
                        '原、被告婚后感情一般，后发现被告生活不检点和异性发生关系，被原告堵其家中，被告染有恶习，谎言连篇，到处借钱。'
                        '借酒发疯打人、骂街、砸东西，经常不回家，对孩子不尽抚养义务，于1999年10月10日起诉红桥法院，法院传唤被告，他无故不到庭，'
                        '所以驳回原告诉求。自1999年7月20日至今没见被告，未同居生活长达19年，诉求法院准予离婚。以准权益。')
    print('end')
    pass