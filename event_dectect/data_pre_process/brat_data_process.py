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
def formLabelData(labelFilePath,savePath,segmentor_model_path,segmentor_user_dict_path,pos_model_path,stop_words_path,trigger_labels_path,argu_labels_path,mode=1):
    if(mode==1):
        savePath = os.path.join(savePath,'Spe')
    else:
        savePath = os.path.join(savePath,'Full')
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
                if (mode == 1):
                    handlerSingleFile(newPath)
                elif (mode == 2):
                    handlerSingleFile2(newPath)

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
                        # label = 'B_' + entity.getType()
                        label = 'B_' + entity.getName()
                        if (label not in labedWords):  # 如果不是关注集里的标注类型，则设为O
                            label = 'O'
                        labeled[index] = label
                        isBegin = False
                    else:
                        # label = 'I_' + entity.getType()
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

    #一个句子可以标注多个事件
    def handlerSingleFile2(filePath):
        if (filePath.find('.ann') == -1):
            return
        # 查看源文件是否存在，如果不存在直接跳过
        originFile = os.path.join(filePath, filePath.replace('.ann', '.txt'))
        if (not os.path.exists(originFile)):
            return

        events = getEvents(filePath,originFile)

        #在源文件上标注标签
        #获取源文件，并将换行符置为2个位置形式
        with open(originFile,'r',encoding='utf8') as f:
            content = f.read()
            content = content.replace('\n','\r\n')
        #分词，构造标签list
        content = list(segmentor.segment(content))
        tags = list(map(lambda x:'O'if(x!='\r\n') else x,content))
        posTags = list(postagger.postag(content))

        #针对每个事件将标签填入
        for event in events:
            #此时传入整个原文，起始索引为0
            labelAEntity(content,tags,event.getTrigger(),0)
            for argument in event.getArguments():
                labelAEntity(content,tags,argument,0)

        #去停用词
        content,tags,posTags=delStopWords(content,tags,posTags)
        newContent = []
        newTags = []
        newPosTags = []
        #在全文里面要保留\n区别换行
        for word,tag,posTag in zip(content,tags,posTags):
            if(word=='\r'): #考虑\r\n被分词的情况，如果\r单独存在，则直接舍掉，
                continue
            if(word.find('\r')!=-1):#如果\r和其他词在一起，则只去掉\r，包括\r\n在一起的情况
                word = word.replace('\r','')
                tag = tag.replace('\r','')
                posTag = posTag.replace('\r','')
            newContent.append(word)
            newTags.append(tag)
            newPosTags.append(posTag)
        # 存储
        theSavePath = ''
        if (filePath.find('qsz') != -1):
            theSavePath = os.path.join(savePath, 'qsz_' + os.path.basename(filePath).replace('.ann', '.txt'))
        if (filePath.find('cpws') != -1):
            theSavePath = os.path.join(savePath, 'cpws' + os.path.basename(filePath).replace('.ann', '.txt'))
        if (filePath.find('qstsbl') != -1):
            theSavePath = os.path.join(savePath, 'qstsbl' + os.path.basename(filePath).replace('.ann', '.txt'))

        #写入同样要按照一行原文，一行标注的格式，如果整行都没有标注，则删去
        with open(theSavePath, 'w', encoding='utf8') as fw:
            wordsLine = []
            tagsLine = []
            posTagsLine = []
            hasTag = False
            for word,tag,posTag in zip(newContent,newTags,newPosTags):
                if (word.find('\n')!=-1):
                    if (hasTag and len(wordsLine)>1): #有可能去停用词之后一行没有内容了，只剩一个\n，这是不用再写入
                        fw.write(' '.join(wordsLine))
                        fw.write('\n')
                        fw.write(' '.join(tagsLine))
                        fw.write('\n')
                        fw.write(' '.join(posTagsLine))
                        fw.write('\n')
                    hasTag= False
                    wordsLine = []
                    tagsLine = []
                    posTagsLine = []
                    continue
                wordsLine.append(word)
                tagsLine.append(tag)
                posTagsLine.append(posTag)
                if(tag !='O' and tag!='<pad>' and tag !='\n'): #如果是全为O的行去掉
                    hasTag = True



    if(os.path.isdir(labelFilePath)):
        handlderDir(labelFilePath)
    else:
        if(mode==1):
            handlerSingleFile(labelFilePath)
        elif(mode==2):
            handlerSingleFile2(labelFilePath)


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
    base_path = 'C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF'
    # base_path = '/root/lstm_crf/data'
    brat_base_path = os.path.join(base_path, 'brat')
    ltp_path = os.path.join(base_path, 'ltp_data_v3.4.0')
    formLabelData(
        labelFilePath=os.path.join(brat_base_path, 'labeled'),
        savePath=os.path.join(base_path, 'labeled'),
        segmentor_model_path=os.path.join(ltp_path, 'cws.model'),
        segmentor_user_dict_path=os.path.join(ltp_path, 'userDict.txt'),
        pos_model_path=os.path.join(ltp_path, 'pos.model'),
        stop_words_path=os.path.join(base_path, 'newStopWords.txt'),
        # trigger_labels_path=os.path.join(base_path,'triggerLabels.txt'),
        # argu_labels_path=os.path.join(base_path,'argumentLabels.txt'),
        trigger_labels_path=os.path.join(base_path,'full_trigger_labels.txt'),
        argu_labels_path=os.path.join(base_path,'full_argu_labels.txt'),
        #mode=1 Spe
        #mode=2 Full
        mode=1)


#将spe模型下的单句合并为full下的句子
def merge(path):
    #会用到trigger集合，需要初始化Trigger_Tags
    parse = getParser()
    CONFIG.init(parse.root_dir)
    #新文件的保存路径，保存在传入文件的同级目录
    savePath = os.path.join(os.path.split(path)[0],'Merge_'+os.path.split(path)[1])
    if(not os.path.exists(savePath)):
        os.mkdir(savePath)

    def merge(tagsList):
        mergedTags = tagsList[0]
        for tags in tagsList[1:]:
            for index,tag in enumerate(tags):
                if(tag!='O'):
                    if(mergedTags[index]=='O'):
                        mergedTags[index] = tag
                    elif(mergedTags[index] in CONFIG.TRIGGER_TAGs):
                        '''此时产生冲突'''
                        '''原先填入的是触发词'''
                        if(mergedTags[index].find('B_')==-1 and tag.find('B_')!=-1):
                            mergedTags[index] = tag #原先的不是B_开头触发词，新来的是B_开头触发词才能覆盖
                    else:#如果以前不是触发词，
                        if((tag in CONFIG.TRIGGER_TAGs or tag.find('B_')!=-1) and mergedTags[index].find('B_')==-1): #只有新来的是触发词或者B_开头的参数，而且老的不是B_开头才能覆盖
                            mergedTags[index] = tag
        return mergedTags

    for fileName in os.listdir(path):
        with open(os.path.join(path,fileName),'r',encoding='utf8') as f,open(os.path.join(savePath,fileName),'w',encoding='utf8') as fw:
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
                    mergedTags = merge(lastTagsList)
                    fw.write(lastSentence+'\n'+' '.join(mergedTags)+'\n'+poses+'\n')
                    #更新缓存
                    lastSentence = sentence
                    lastTagsList = []
                    lastTags = f.readline().strip().split()
                    lastTagsList.append(lastTags)
                    poses = f.readline().strip()

                    sentence = f.readline().strip()

            #处理缓存
            mergedTags = merge(lastTagsList)
            fw.write(lastSentence+'\n'+' '.join(mergedTags)+'\n'+poses+'\n')

if __name__ == '__main__':
    # merge('C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\labeled\\Spe')
    main()
    print ('end')
    sys.exit(0)
