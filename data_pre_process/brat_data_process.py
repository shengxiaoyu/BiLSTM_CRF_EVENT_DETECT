#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__doc__ = 'description将brat生成的ann文件和源文件.txt结合，生成人工标注的样子的文件'
__author__ = '13314409603@163.com'


import os
import sys
from pyltp import Segmentor
#将案号转为indxe 文件名称,brat文件名不能含中文
def an2Index(path):
    if(not os.path.exists(path)):
        return
    elif(os.path.isdir(path)):
        for file in os.listdir(path):
            an2Index(os.path.join(path,file))
    else:
        global COUNT
        os.renames(path,os.path.join(os.path.dirname(path),str(COUNT)+'.txt'))
        COUNT += 1

#获得所有标注类型
def getLabels(bratPath):
    labels = []
    configPath = os.path.join(os.path.join(bratPath,'config'),'annotation.conf')
    isEntity = '1'
    isEvent = '2'
    mode = None
    with open(configPath,'r',encoding='utf8') as f:
        for line in f.readlines():
            if(len(line.strip())==0 or line.strip().startswith('#')):
                continue
            if(line.startswith('[entities]')):
                mode = isEntity
                continue
            if(line.startswith('[relations]') or line.startswith('[attributes]')):
                mode = None
                continue
            if(line.startswith('[events]')):
                mode = isEvent
                continue

            #参数
            if(mode == isEntity):
                labels.append(line.strip())

            #触发词
            elif(mode == isEvent):
                if(line.startswith('\t')):
                    labels.append(line.strip().split('\t')[0])
    with open(os.path.join(bratPath,'labels.txt'),'w',encoding='utf8') as fw:
        fw.write('O\n')
        newLabels = []
        BIList =list(map(lambda x:['B_'+x,'I_'+x],labels))
        list(map(lambda x:newLabels.extend(x),BIList))

        fw.write('\n'.join(newLabels))


#获取触发词集 从.ann文件夹
def getTriggerSet(path,events_triggers):
    #处理单个文件
    def handleSingleFile(path,events_triggers):
        event_entitys = []
        entitys = {}
        print(path)
        with open(path,'r',encoding='utf8') as f:
            for line in f.readlines():
                # print(line)
                if(line.startswith('T')):
                    entity = Entity(line)
                    entitys[entity.id] = entity
                elif(line.startswith('E')):
                    event_entitys.append(Relation(line))
        for event_entity in event_entitys:
            triggerId = event_entity.getParameters()[0][1]
            trigger = entitys.get(triggerId)
            triggers = events_triggers.get(trigger.getType())
            if(triggers == None):
                triggers = set([])
            triggers.add(trigger.getValue())
            events_triggers[trigger.getType()] = triggers
    if(os.path.isdir(path)):
        for fileName in os.listdir(path):
            getTriggerSet(os.path.join(path,fileName),events_triggers)
    elif(path.endswith('.ann')):
        handleSingleFile(path,events_triggers)
#将触发词写入文件
def writeTriggerToFile(events_triggers,savePath):
    savePath = os.path.join(savePath,'triggers')
    if(not os.path.exists(savePath)):
        os.mkdir(savePath)
    for event_triggers in events_triggers.items():
        oneEventPath = os.path.join(savePath,event_triggers[0]+'.txt')
        newTriggers = event_triggers[1]
        if(os.path.exists(oneEventPath)):
            with open(oneEventPath,'r',encoding='utf8') as f:
                oldTriggers = set(f.read().strip().split('\n'))
                newTriggers = newTriggers.union(oldTriggers)
        with open(os.path.join(savePath,event_triggers[0]+'.txt'),'w',encoding='utf8') as f:
            for trigger in event_triggers[1]:
                f.write(trigger+'\n')


#将源文件和标注文件合一
def formLabelData(labelFilePath,savePath,segmentor_model_path,segmentor_user_dict_path,stop_words_path):
    # 分词器
    segmentor = Segmentor()
    segmentor.load_with_lexicon(segmentor_model_path, segmentor_user_dict_path)

    def handlderDir(dirPath):
        for fileName in os.listdir(dirPath):
            newPath = os.path.join(dirPath, fileName)
            if (os.path.isdir(newPath)):
                handlderDir(newPath)
            else:
                handlerSingleFile(newPath)

    def handlerSingleFile(filePath):
        if (filePath.find('.ann') == -1):
            return
        # 查看源文件是否存在，如果不存在直接跳过
        originFile = os.path.join(filePath, filePath.replace('.ann', '.txt'))
        if (not os.path.exists(originFile)):
            return

        # 读取ann文件，获取标注记录
        relations = []  # 存储事件和参数关系Relation
        entitiesDict = {}  # 存储参数实体Entity
        with open(filePath, 'r', encoding='utf8') as fLabel:
            for line in fLabel.readlines():
                if (line.startswith('T')):
                    entity = Entity(line)
                    entitiesDict[entity.getId()] = entity
                if (line.startswith('E')):
                    relations.append(Relation(line))

        events = []  # 存储事件

        # 根据初始化的relations和entitiesDict完善entites的name，构造event
        for relation in relations:
            event = None
            for index, paramter in enumerate(relation.getParameters()):  # 形如[['Marry','T3'],['Time','T1']]
                if (index == 0):  # 第一个是描述事件触发词的entity
                    # 构造事件object
                    event = Event(relation.id, paramter[0])
                    # 获得触发词对应的entity
                    entity = entitiesDict.get(paramter[1])
                    # 设置触发词的名称：事件类型_Trigger
                    entity.setName(paramter[0] + '_Trigger')
                    # 填入触发词
                    event.setTrigger(entity)
                else:
                    # 事件参数处理
                    entity = entitiesDict.get(paramter[1])
                    entity.setName(event.getType() + '_' + paramter[0])
                    event.addArgument(entity)
            events.append(event)

        # 将事件按标注索引最小位排序
        events.sort(key=lambda x: x.getBegin())

        # 把每个事件涉及的原文语句填入
        for event in events:
            # eventBeginLineIndex = 0
            with open(originFile, 'r', encoding='utf8') as fData:
                cursor = 0
                for line in fData.readlines():
                    line = line.replace('\n', '\r\n')
                    beginIndexOfTheLine = cursor
                    endIndexOfTheLine = cursor + len(line)

                    # 标注起止范围都在在当前句子内
                    if (endIndexOfTheLine <= event.beginIndex):
                        cursor = endIndexOfTheLine
                        continue
                    if (beginIndexOfTheLine <= event.beginIndex and event.beginIndex <= endIndexOfTheLine
                            and beginIndexOfTheLine <= event.endIndex and event.endIndex <= endIndexOfTheLine):
                        event.addSentence(line)
                        event.setBeginLineIndex(beginIndexOfTheLine)
                        break
                    # 只有起始范围在当前句子
                    elif (beginIndexOfTheLine <= event.beginIndex and event.beginIndex <= endIndexOfTheLine and
                          endIndexOfTheLine < event.endIndex):
                        event.addSentence(line)
                        event.setBeginLineIndex(beginIndexOfTheLine)
                        # 只有截止范围在当前句子
                    elif (event.beginIndex < beginIndexOfTheLine and beginIndexOfTheLine <= event.endIndex and
                          event.endIndex <= endIndexOfTheLine):
                        event.addSentence(line)
                        break
                    cursor = endIndexOfTheLine

        # 把每个事件涉及的原句分词并标注
        for event in events:
            def labelAEntity(words, labeled, entity, baseIndex):
                coursor = baseIndex
                isBegin = True
                for index, word in enumerate(words):
                    beginCoursor = coursor
                    endCoursor = len(word) + coursor
                    if ((beginCoursor <= entity.getBegin() and entity.getBegin() < endCoursor) or
                            (beginCoursor < entity.getEnd() and entity.getEnd() <= endCoursor) or
                            (beginCoursor >= entity.getBegin() and endCoursor <= entity.getEnd())):
                        # 此时待标记entity进入范围，
                        # 考虑标签和分词不对应的情况，一个词被对应到多次标记，因为先标记触发词，所以优先级第一，其余的越靠后越低
                        if (labeled[index].find('O') != -1):
                            if(isBegin):
                                labeled[index] = 'B_'+entity.getType()
                                isBegin = False
                            else:
                                labeled[index] = 'I_'+entity.getType()

                    coursor = endCoursor

            words = list(segmentor.segment(''.join(event.getSentences())))
            tags = list(map(lambda x: 'O' if (x != '\r\n') else x, words))
            labelAEntity(words, tags, event.getTrigger(), event.getBeginLineIndex())
            for argument in event.getArguments():
                labelAEntity(words, tags, argument, event.getBeginLineIndex())
            event.setWords(words)
            event.setTags(tags)

        # 去停用词
        with open(stop_words_path, 'r', encoding='utf8') as f:
            stopWords = set(f.read().split())
        for event in events:
            newWords = []
            newTags = []
            for word, tag in zip(event.getWords(), event.getTags()):
                if (word not in stopWords and word!='\r\n'):
                    newWords.append(word)
                    newTags.append(tag)
            if(len(newWords)!=len(newTags)):
                print("error")
            event.setTags(newTags)
            event.setWords(newWords)

        # 存储
        theSavePath = ''
        if(filePath.find('qsz')!=-1):
            theSavePath = os.path.join(savePath,'qsz_'+os.path.basename(filePath).replace('.ann', '.txt'))
        if (filePath.find('cpws') != -1):
            theSavePath = os.path.join(savePath,'cpws'+os.path.basename(filePath).replace('.ann', '.txt'))
        if (filePath.find('qstsbl') != -1):
            theSavePath = os.path.join(savePath,'qstsbl'+os.path.basename(filePath).replace('.ann', '.txt'))
        with open(theSavePath, 'w', encoding='utf8') as fw:
            for event in events:
                fw.write(' '.join(event.getWords()))
                fw.write('\n')
                fw.write(' '.join(event.getTags()))
                fw.write('\n')

    if(os.path.isdir(labelFilePath)):
        handlderDir(labelFilePath)
    else:
        handlerSingleFile(labelFilePath)

    segmentor.release()

#构造停用词表，否定词不能作为停用词去掉
def stopWords(path):
    stopWords = set()
    stopPath = os.path.join(base_path,'stopWords')
    for file in os.listdir(stopPath):
        with open(os.path.join(stopPath,file),'r',encoding='utf8') as f:
            content = f.read()
            stopWords = stopWords.union(set(content.split('\n')))
    negativePath = os.path.join(base_path,'negativeWords')
    with open(os.path.join(negativePath,'dict_negative.txt'),'r',encoding='utf8') as f:
        negativeWords = set(map(lambda line:line.split('\t')[0],f.readlines()))
    stopWords = stopWords.difference(negativeWords)
    with open(os.path.join(path,'newStopWords.txt'),'w',encoding='utf8') as fw:
        fw.write('\n'.join(stopWords))

#记录一个标注体，形如T1   Person 17 19    双方
#表示标注体ID为T1，标注体类型为Person，标注范围为[17,19)，标注的值为“双方”
class Entity(object):
    def __init__(self,str):
        splits = str.strip().split('\t')
        self.id = splits[0]
        self.type = splits[1].split()[0] #参数类型，比如Person
        self.beginIndex = int(splits[1].split()[1])
        self.endIndex = int(splits[1].split()[2])
        self.value = splits[2]
        self.name = None #参数在具体事件中的名称，比如Participant_Person
    def getId(self):
        return self.id
    def getBegin(self):
        return self.beginIndex
    def getEnd(self):
        return self.endIndex
    def setName(self,str):
        self.name = str
    def getValue(self):
        return self.value
    def getName(self):
        return self.name
    def getType(self):
        return self.type

class Event(object):
    def __init__(self,id,type):
        self.id = id
        self.type = type
        self.arguments = []
        self.trigger = None

        #该事件标注索引最小最大位
        self.beginIndex = sys.maxsize
        self.endIndex = -1
        self.sentence = []
        self.beginLineIndex = 0 #该事件在原文本中涉及范围第一行的起始索引

        self.words = []
        self.tags = []
    def addSentence(self,sentence):
        self.sentence.append(sentence)
    def setType(self,type):
        self.type = type
    def getType(self):
        return self.type
    def setTrigger(self,entity):
        self.trigger = entity
        self.beginIndex = min(self.beginIndex,entity.getBegin())
        self.endIndex = max(self.endIndex,entity.getEnd())
    def getTrigger(self):
        return self.trigger
    def addArgument(self,entity):
        self.arguments.append(entity)
        self.beginIndex = min(self.beginIndex,entity.getBegin())
        self.endIndex = max(self.endIndex,entity.getEnd())
    def getArguments(self):
        return self.arguments
    def getBegin(self):
        return self.beginIndex
    def getEnd(self):
        return self.endIndex
    def setBeginLineIndex(self,index):
        self.beginLineIndex = index
    def getBeginLineIndex(self):
        return self.beginLineIndex
    def getSentences(self):
        return self.sentence
    def setTags(self,tags):
        self.tags = tags
    def getTags(self):
        return self.tags
    def setWords(self,words):
        self.words = words
    def getWords(self):
        return self.words
# 记录一个事件的关系，源数据形如：E1	Marry:T2 Time:T3 Participant:T1
# 表示事件Marry:T2,有参数Time:T3和Participant:T1
class Relation(object):
    def __init__(self,str):
        splits = str.split('\t')
        self.id = splits[0]
        self.parameters = list(map(lambda str:str.split(':'),splits[1].split())) #[[Marray,T2].[Time,T3}...]
    def getParameters(self):
        return self.parameters

if __name__ == '__main__':
    base_path = 'C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF'
    brat_base_path = os.path.join(base_path,'brat')
    ltp_path = os.path.join(base_path,'ltp_data_v3.4.0')
    formLabelData(
                  os.path.join(brat_base_path,'labeled'),
                  os.path.join(base_path,'labeled'),
                  os.path.join(ltp_path,'cws.model'),
                  os.path.join(ltp_path,'userDict.txt'),
                  os.path.join(base_path,'newStopWords.txt'))
    #
    # # stopWords(base_path)
    #
    # getLabels(brat_base_path)
    # events_triggers = dict()
    # getTriggerSet(os.path.join(brat_base_path,'labled'),events_triggers)
    # writeTriggerToFile(events_triggers,brat_base_path)
    print ('end')
    sys.exit(0)
    pass
