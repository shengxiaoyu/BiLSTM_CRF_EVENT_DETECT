#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from config_parser import getParser

__doc__ = 'description将brat生成的ann文件和源文件.txt结合，生成人工标注的样子的文件'
__author__ = '13314409603@163.com'


import os
import sys
from pyltp import Segmentor
from pyltp import Postagger
from event_dectect.First_For_Commo_Tags import config_center as CONFIG


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
def formLabelData(labelFilePath,savePath,segmentor_model_path,segmentor_user_dict_path,pos_model_path,stop_words_path,trigger_labels_path,argu_labels_path):
    savePath = os.path.join(savePath,'Spe')

    # 分词器
    segmentor = Segmentor()
    segmentor.load_with_lexicon(segmentor_model_path, segmentor_user_dict_path)
    postagger = Postagger()
    postagger.load(pos_model_path)
    #停用词集
    with open(stop_words_path, 'r', encoding='utf8') as f:
        stopWords = set(f.read().split())
    #关注标签集
    with open(trigger_labels_path, 'r', encoding='utf8') as f,open(argu_labels_path,'r',encoding='utf8') as f2:
        labedWords = set(f.read().split())
        for line in f2.readlines():
            labedWords.add(line.strip())
    eventsType = {}
    def handlderDir(dirPath):
        for fileName in os.listdir(dirPath):
            newPath = os.path.join(dirPath, fileName)
            if (os.path.isdir(newPath)):
                handlderDir(newPath)
            else:
                handlerSingleFile(newPath)

    #获取event列表，其中包括每个事件含有的原文句子
    def getEvents(filePath,originFile):
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
                    if (paramter[0] == 'Negation'):
                        entity.setName(paramter[0])
                        # neg.add(entity.getValue())
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
        return events

        # 标注单个事件参数或者trigger触发词
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
                    if (isBegin):
                        label = 'B_' + entity.getName()
                        if (label not in labedWords):  # 如果不是关注集里的标注类型，则设为O
                            label = 'O'
                        labeled[index] = label
                        isBegin = False
                    else:
                        label = 'I_' + entity.getName()
                        if (label not in labedWords):  # 如果不是关注集里的标注类型，则设为O
                            label = 'O'
                        labeled[index] = label
            coursor = endCoursor

        #去停用词
    def delStopWords(words,tags,postags):
        newWords=[]
        newTags=[]
        newPosTags=[]
        ifDropBegin = False
        for word,tag,pos in zip(words,tags,postags):
            if (word not in stopWords):
                newWords.append(word)
                if(ifDropBegin):
                    newTags.append(tag.replace('I_','B_'))
                    ifDropBegin = False
                else:
                    newTags.append(tag)
                newPosTags.append(pos)
            else:#考虑如果把B_作为开始标志的词去掉时，需要把下一个I_改为B_
                if(tag.find('B_')!=-1):
                    ifDropBegin = True
        return newWords,newTags,newPosTags

    # 分割成一个句子只标注一个事件
    def handlerSingleFile(filePath):
        if (filePath.find('.ann') == -1):
            return
        # 查看源文件是否存在，如果不存在直接跳过
        originFile = os.path.join(filePath, filePath.replace('.ann', '.txt'))
        if (not os.path.exists(originFile)):
            return
        print(filePath)
        #获取事件list
        events = getEvents(filePath,originFile)

        # 把每个事件涉及的原句分词并标注
        for event in events:
            words = list(segmentor.segment(''.join(event.getSentences())))
            tags = list(map(lambda x: 'O' if (x != '\r\n') else x, words))
            posTags = list(postagger.postag(words))
            labelAEntity(words, tags, event.getTrigger(), event.getBeginLineIndex())
            for argument in event.getArguments():
                labelAEntity(words, tags, argument, event.getBeginLineIndex())
            if(len(words)!=len(tags) or len(words)!=len(posTags)):
                print(filePath+': \n'+words+'\n'+tags)

            #去停用词
            words,tags,posTags = delStopWords(words,tags,posTags)
            event.setWords(words)
            event.setTags(tags)
            event.setPosTags(posTags)


        # 去换行符
        for event in events:
            newWords = []
            newTags = []
            newPosTags = []
            for word, tag,posTag in zip(event.getWords(), event.getTags(),event.getPosTags()):
                if (word.find('\n') != -1 or word.find('\r') != -1):
                    continue
                else:
                    newWords.append(word)
                    newTags.append(tag)
                    newPosTags.append(posTag)

            if (len(newWords) != len(newTags)):
                print("error")
            event.setTags(newTags)
            event.setWords(newWords)
            event.setPosTags(newPosTags)
            eventType = event.getType()
            if (eventType in eventsType):
                eventsType[eventType] += 1
            else:
                eventsType[eventType] = 1

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
                fw.write(' '.join(event.getPosTags()))
                fw.write('\n')
                # if('B_Wealth_Value' in event.getTags() or 'B_Debt_Value' in event.getTags() or 'B_Credit_Value' in event.getTags()):
                #     w1.write(' '.join(event.getWords())+'\n')

    w1 = open(r'C:\Users\13314\Desktop\test\values_Str.txt','w',encoding='utf8')
    # values = set()
    if(os.path.isdir(labelFilePath)):
        handlderDir(labelFilePath)
    else:
        handlerSingleFile(labelFilePath)


    segmentor.release()
    print(eventsType)

#构造停用词表，否定词不能作为停用词去掉
def stopWords(base_path):
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
    with open(os.path.join(base_path,'newStopWords.txt'),'w',encoding='utf8') as fw:
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
        self.beginLineIndex = 0 #该事件在原文本中涉及范围第一行的起始索引,因为将每个事件涉及的句子单独提出来之后，去标标签时需要指定这个句子在原文中的起始索引

        self.words = []
        self.tags = []
        self.posTags = []
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
    def setPosTags(self,posTags):
        self.posTags = posTags
    def getPosTags(self):
        return self.posTags
# 记录一个事件的关系，源数据形如：E1	Marry:T2 Time:T3 Participant:T1
# 表示事件Marry:T2,有参数Time:T3和Participant:T1
class Relation(object):
    def __init__(self,str):
        splits = str.split('\t')
        self.id = splits[0]
        self.parameters = list(map(lambda str:str.split(':'),splits[1].split())) #[[Marray,T2].[Time,T3}...]
    def getParameters(self):
        return self.parameters
def main():
    base_path = r'A:\Bi-LSTM+CRF'
    # base_path = '/root/lstm_crf/data'
    brat_base_path = os.path.join(base_path, 'brat')
    ltp_path = os.path.join(base_path, 'ltp_data_v3.4.0')
    #生成标注样例，每个样例一个事件，标签为最终标签
    formLabelData(
        labelFilePath=os.path.join(brat_base_path, 'relabled'),
        savePath=os.path.join(base_path, 'labeled'),
        segmentor_model_path=os.path.join(ltp_path, 'cws.model'),
        segmentor_user_dict_path=os.path.join(ltp_path, 'userDict.txt'),
        pos_model_path=os.path.join(ltp_path, 'pos.model'),
        stop_words_path=os.path.join(base_path, 'newStopWords.txt'),
        trigger_labels_path=os.path.join(base_path,'full_trigger_labels.txt'),
        # trigger_labels_path=os.path.join(base_path,'triggerLabels.txt'),
        argu_labels_path=os.path.join(base_path,'full_argu_labels.txt'),
        # argu_labels_path=os.path.join(base_path,'argumentLabels.txt')
        )


#将spe模型下的单句合并为full下的句子
secondeTag_2_firstTag = {
    'B_Know_Time':'B_Time',
    'I_Know_Time':'I_Time',
    'B_BeInLove_Time':'B_Time',
    'I_BeInLove_Time':'I_Time',
    'B_Marry_Time':'B_Time',
    'I_Marry_Time':'I_Time',
    'B_Remarry_Participant':'B_Person',
    'I_Remarry_Participant':'I_Person',
    'B_Bear_DateOfBirth':'B_Time',
    'I_Bear_DateOfBirth':'I_Time',
    'B_Bear_Gender':'B_Gender',
    'I_Bear_Gender':'I_Gender',
    'B_Bear_ChildName':'B_Name',
    'I_Bear_ChildName':'I_Name',
    'B_Bear_Age':'B_Age',
    'I_Bear_Age':'I_Age',
    'B_DomesticViolence_Perpetrators':'B_Person',
    'I_DomesticViolence_Perpetrators':'I_Person',
    'B_DomesticViolence_Victim':'B_Person',
    'I_DomesticViolence_Victim':'I_Person',
    'B_DomesticViolence_Time':'B_Time',
    'I_DomesticViolence_Time':'I_Time',
    'B_BadHabit_Participant':'B_Person',
    'I_BadHabit_Participant':'I_Person',
    'B_Derailed_Time':'B_Time',
    'I_Derailed_Time':'I_Time',
    'B_Derailed_Derailer':'B_Person',
    'I_Derailed_Derailer':'I_Person',
    'B_Separation_BeginTime':'B_Time',
    'I_Separation_BeginTime':'I_Time',
    'B_Separation_EndTime':'B_Time',
    'I_Separation_EndTime':'I_Time',
    'B_Separation_Duration':'B_Duration',
    'I_Separation_Duration':'I_Duration',
    'B_DivorceLawsuit_SueTime':'B_Time',
    'I_DivorceLawsuit_SueTime':'I_Time',
    'B_DivorceLawsuit_Initiator':'B_Person',
    'I_DivorceLawsuit_Initiator':'I_Person',
    'B_DivorceLawsuit_Court':'B_Court',
    'I_DivorceLawsuit_Court':'I_Court',
    'B_DivorceLawsuit_Result':'B_Judgment',
    'I_DivorceLawsuit_Result':'I_Judgment',
    'B_DivorceLawsuit_JudgeTime':'B_Time',
    'I_DivorceLawsuit_JudgeTime':'I_Time',
    'B_DivorceLawsuit_JudgeDocument':'B_Document',
    'I_DivorceLawsuit_JudgeDocument':'I_Document',
    'B_Wealth_Value':'B_Price',
    'I_Wealth_Value':'I_Price',
    'B_Wealth_IsPersonal':'B_PersonalProperty',
    'I_Wealth_IsPersonal':'I_PersonalProperty',
    'B_Wealth_Whose':'B_Person',
    'I_Wealth_Whose':'B_Person',
    'B_Wealth_IsCommon':'B_CommonProperty',
    'I_Wealth_IsCommon':'I_CommonProperty',
    'B_Debt_Creditor':'B_Person',
    'I_Debt_Creditor':'I_Person',
    'B_Debt_Value':'B_Price',
    'I_Debt_Value':'I_Price',
    'B_Credit_Debtor':'B_Person',
    'I_Credit_Debtor':'I_Person',
    'B_Credit_Value':'B_Price',
    'I_Credit_Value':'I_Price',
    'B_Know_Trigger':'B_Know',
    'I_Know_Trigger':'I_Know',
    'B_BeInLove_Trigger':'B_BeInLove',
    'I_BeInLove_Trigger':'I_BeInLove',
    'B_Marry_Trigger':'B_Marry',
    'I_Marry_Trigger':'I_Marry',
    'B_Remarry_Trigger':'B_Remarry',
    'I_Remarry_Trigger':'I_Remarry',
    'B_Bear_Trigger':'B_Bear',
    'I_Bear_Trigger':'I_Bear',
    'B_FamilyConflict_Trigger':'B_FamilyConflict',
    'I_FamilyConflict_Trigger':'I_FamilyConflict',
    'B_DomesticViolence_Trigger':'B_DomesticViolence',
    'I_DomesticViolence_Trigger':'I_DomesticViolence',
    'B_BadHabit_Trigger':'B_BadHabit',
    'I_BadHabit_Trigger':'I_BadHabit',
    'B_Derailed_Trigger':'B_Derailed',
    'I_Derailed_Trigger':'I_Derailed',
    'B_Separation_Trigger':'B_Separation',
    'I_Separation_Trigger':'I_Separation',
    'B_DivorceLawsuit_Trigger':'B_DivorceLawsuit',
    'I_DivorceLawsuit_Trigger':'I_DivorceLawsuit',
    'B_Wealth_Trigger':'B_Wealth',
    'I_Wealth_Trigger':'I_Wealth',
    'B_Debt_Trigger':'B_Debt',
    'I_Debt_Trigger':'I_Debt',
    'B_Credit_Trigger':'B_Credit',
    'I_Credit_Trigger':'I_Credit',
    'B_Negation': 'B_Negated',
    'I_Negation': 'I_Negated',
    'O':'O',
}


NUM1 = 0
NUM2 = 0
SHARE_EVENT = {}
SHARE_ARGU ={}
#将spe模型下的单句合并为full下的句子
def merge(path):
    #会用到trigger集合，需要初始化Trigger_Tags
    parse = getParser()
    CONFIG.init(parse.root_dir)
    first_save_dir = os.path.join(os.path.split(path)[0],'Merge_for_first')
    second_save_dir = os.path.join(os.path.split(path)[0],'Merge_for_second')
    dirs = ['03','36','69']
    for dir in dirs:
        for i in range(1,11):
            current_path = os.path.join(os.path.join(path,dir),str(i))
            #新文件的保存路径，保存在传入文件的同级目录
            #第一个模型的数据
            savePath = os.path.join(os.path.join(first_save_dir,dir),str(i))
            os.makedirs(savePath)
            #第二个模型的数据保存卢坚
            savePath2 = os.path.join(os.path.join(second_save_dir,dir),str(i))
            os.makedirs(savePath2)

            if(not os.path.exists(savePath)):
                os.mkdir(savePath)
            if (not os.path.exists(savePath2)):
                os.mkdir(savePath2)
            for fileName in os.listdir(current_path):
                with open(os.path.join(current_path,fileName),'r',encoding='utf8') as f,open(os.path.join(savePath,fileName),'w',encoding='utf8') as fw,\
                        open(os.path.join(savePath2,fileName),'w',encoding='utf8') as fw2:
                    #words行
                    lastSentence = f.readline().strip()
                    #tag行
                    lastTagsList = []
                    lastTags = f.readline().strip().split()
                    lastTagsList.append(lastTags)
                    #pos行
                    poses = f.readline().strip()

                    sentence = f.readline().strip()
                    while(sentence):
                        if(sentence==lastSentence):
                            '''此时时同一行，将tags加入列表'''
                            lastTagsList.append(f.readline().strip().split())
                            #去掉pos行
                            f.readline()
                            #更新words行
                            sentence = f.readline().strip()
                        else:
                            '''来了新的行，将上一种合并写入'''
                            mergedTags = one_merge(lastTagsList)

                            #更新，生成文件格式：一行原句，一行通用标签，一行pos
                            fw.write(lastSentence+'\n'+' '.join(mergedTags)+'\n'+poses+'\n')
                            #更新，生成文件格式：一行原句，一行通用标签，一行pos，一行细粒度标签
                            for lastTags in lastTagsList:
                                fw2.write(lastSentence+'\n'+' '.join(mergedTags)+'\n'+poses+'\n'+' '.join(lastTags)+'\n')
                            #更新缓存
                            lastSentence = sentence
                            lastTagsList = []
                            lastTags = f.readline().strip().split()
                            lastTagsList.append(lastTags)
                            poses = f.readline().strip()

                            sentence = f.readline().strip()

                    #处理缓存
                    mergedTags = one_merge(lastTagsList)
                    fw.write(lastSentence+'\n'+' '.join(mergedTags)+'\n'+poses+'\n')
                    for lastTags in lastTagsList:
                        fw2.write(lastSentence + '\n' + ' '.join(mergedTags) + '\n' + poses + '\n' + ' '.join(lastTags) + '\n')
            print(NUM1)
            print(NUM2)
            print(SHARE_EVENT)


def one_merge(tagsList):
    global NUM1,NUM2,SHARE_EVENT,SHARE_ARGU
    mergedTags = ['O' for _ in range(len(tagsList[0]))]
    for tags in tagsList:
        for index,tag in enumerate(tags):
            if(tag!='O'):
                if(mergedTags[index]=='O'):
                    mergedTags[index] = secondeTag_2_firstTag[tag]
                    # mergedTags[index] = tag
                elif(mergedTags[index] in CONFIG.TRIGGER_TAGs):
                    '''此时产生冲突'''
                    '''原先填入的是触发词'''
                    if(mergedTags[index].find('B_')==-1 and tag.find('B_')!=-1):
                        mergedTags[index] = secondeTag_2_firstTag[tag] #原先的不是B_开头触发词，新来的是B_开头触发词才能覆盖
                else:#如果以前不是触发词，
                    if((tag in CONFIG.TRIGGER_TAGs or tag.find('B_')!=-1) and mergedTags[index].find('B_')==-1): #只有新来的是触发词或者B_开头的参数，而且老的不是B_开头才能覆盖
                        mergedTags[index] = secondeTag_2_firstTag[tag]
    return mergedTags

if __name__ == '__main__':
    main()
    # merge(r'A:\Bi-LSTM+CRF\labeled\Spe')
    print ('end')
    sys.exit(0)
