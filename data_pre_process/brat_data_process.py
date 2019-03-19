#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

__doc__ = 'description将brat生成的ann文件和源文件.txt结合，生成人工标注的样子的文件'
__author__ = '13314409603@163.com'


import os

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




#获取触发词集 从.ann文件夹
def getTriggerSet(path,events_triggers):
    #处理单个文件
    def handleSingleFile(path,events_triggers):
        event_entitys = []
        entitys = {}
        with open(path,'r',encoding='utf8') as f:
            for line in f.readlines():
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
        with open(os.path.join(savePath,event_triggers[0]+'.txt'),'w',encoding='utf8') as f:
            for trigger in event_triggers[1]:
                f.write(trigger+'\n')


#将源文件和标注文件合一
def formLabelData(originFilePath,labelFilePath,savePath,mode='char'):
    for annName in os.listdir(labelFilePath):
        #查看源文件是否存在
        originFile = os.path.join(originFilePath, annName.replace('.ann', '.txt'))
        if (not os.path.exists(originFile)):
            continue

        #先读取ann文件，获取标注记录
        relations = []
        entitiesDict = {}
        entities = []
        with open(os.path.join(labelFilePath,annName), 'r', encoding='utf8') as fLabel:
            for line in fLabel.readlines():
                if (line.startswith('T')):
                    entity = Entity(line)
                    entitiesDict[entity.getId()] = entity
                if (line.startswith('E')):
                    relations.append(Relation(line))

        #根据初始化的relations和entitiesDict完善entites的name，并加入entities
        for relation in relations:
            eventName = None
            for index,paramter in enumerate(relation.getParameters()):
                if(index==0):#第一个是描述事件触发词的entity
                    eventName = paramter[0]
                    entity = entitiesDict.get(paramter[1])
                    entity.setName(eventName+'_trigger')
                    entities.append(entity)
                else:
                    entity = entitiesDict.get(paramter[1])
                    entity.setName(eventName+'_'+paramter[0])
                    entities.append(entity)
        #并按beginIndex排序
        entities.sort(key=lambda x:x.getBegin())

        #读取源文件并将构造完备的标注文件内容
        newConten=''
        with open(originFile,'r',encoding='utf8') as fData:
            fileIndex = 0 #记录文件字符索引位置
            entityIndex = 0 #记录entity在entities中的索引
            line = fData.readline()
            # 标注行,初始化所有词对应的都为O
            labelLine = ['O' for _ in line.split()]
            while(True):#形如：原 被告 双方 系 夫妻 关系
                if(entityIndex<len(entities) and
                        entities[entityIndex].getBegin()-fileIndex>=0 and
                        entities[entityIndex].getBegin()-fileIndex<len(line)):#说明这一行中有标注数据
                    entity = entities[entityIndex]
                    begin = entity.getBegin() - fileIndex
                    end = entity.getEnd() - fileIndex
                    if(end>=len(line)):#默认整个标注体都在一行，不存在一个标注体跨行的情况
                        end = -1
                    value = line[begin:end]
                    assert entity.getValue().startswith(value) #entity里面的value一定是包含实际取到的value
                    firtIndex = len(line[:begin].strip().split()) #标注开始词的索引
                    if(begin!=0 and line[begin-1:begin]!=' '):#有可能标注的头部是属于一个词的部分，比如：名毛 雨泽，此时标注从“毛”开始，这时需要从“名毛”开始标注
                        firtIndex -= 1
                    length = len(value.split()) #要标注几个词
                    for i in range(length):
                        labelLine[i+firtIndex] = entity.getName()
                    entityIndex += 1 #更新标注索引，下一个标注
                else:
                    newConten += line
                    if(not line.endswith('\n')): #最后一行可能末尾没有换行符
                        newConten += '\n'
                    newConten += (' '.join(labelLine) + '\n')  # 将新的标注行加入

                    fileIndex += (len(line)+1) #更新文件字符索引，在brat生成的配置文件中，换行符占2位
                    line = fData.readline()#换行
                    if(not line): #文件结束
                        break
                    labelLine = ['O' for _ in line.split()]
        print(newConten)

        #将新标注文件写入文件：
        with open(os.path.join(savePath,annName.replace('.ann','.txt')),'w',encoding='utf8') as fw:
            fw.write(newConten)


#记录一个标注体，形如T1   Person 17 19    双方
#表示标注体ID为T1，标注体类型为Person，标注范围为[17,19)，标注的值为“双方”
class Entity(object):
    def __init__(self,str):
        splits = str.strip().split('\t')
        self.id = splits[0]
        self.type = splits[1].split()[0]
        self.beginIndex = int(splits[1].split()[1])
        self.endIndex = int(splits[1].split()[2])
        self.value = splits[2]
        self.name = None
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
    def __init__(self,id):
        self.id = id
        self.arguments = []
        self.triiger = None
    def setTrigger(self,entity):
        self.triiger = entity
    def addArgument(self,entity):
        self.arguments.append(entity)

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
    # formLabelData('C:\\Users\\13314\Desktop\\Bi-LSTM+CRF\\segment_result\\起诉状',
    #               'C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\labeled',
    #               'C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\NERdata\\dev', 'word')
    events_triggers = dict()
    brat_base_path = 'C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\brat'
    getTriggerSet(brat_base_path,events_triggers)
    writeTriggerToFile(events_triggers,brat_base_path)
    print ('end')
    sys.exit(0)
    pass
