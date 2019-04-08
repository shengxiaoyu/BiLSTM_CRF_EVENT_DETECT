#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

from Config.config_parser import getParser
__doc__ = 'description'
__author__ = '13314409603@163.com'



def EventFactory(trigger,word,beginIndex,EndIndex):
    eventDict = {
        'Know':Know,
        'BeInLove':BeInLove,
        'Marry':Marray,
        'Remarry':Remarray,
        'Bear':Bear,
        'FamilyConflict':FamilyConflict,
        'DomesticViolence':DomesticViolence,
        'BadHabit':BadHabit,
        'Derailed':Derailed,
        'Separation':Separation,
        'DivorceLawsuit':DivorceLawsuit,
        'Wealth':Wealth,
        'Debt':Debt,
        'Credit':Credit,
    }
    return eventDict[trigger](trigger,word,beginIndex,EndIndex)
    # if(trigger.find('Know')!=-1):
    #     return Know(trigger,word,beginIndex,EndIndex)
    # if(trigger.find('BeInLove')!=-1):
    #     return BeInLove(trigger,word,beginIndex,EndIndex)
    # if(trigger.find('Marry')!=-1):
    #     return Marray(trigger,word,beginIndex,EndIndex)
    # if(trigger.find('Remarry')!=-1):
    #     return Remarray(trigger,word,beginIndex,EndIndex)
    # if(trigger.find('Bear')!=-1):
    #     return Bear(trigger,word,beginIndex,EndIndex)
    # if (trigger.find('FamilyConflict') != -1):
    #     return F(trigger, word, beginIndex, EndIndex)
    # if (trigger.find('Bear') != -1):
    #     return Bear(trigger, word, beginIndex, EndIndex)

class baseModel(object):
    def __init__(self,trigger,word,beginIndex,endIndex):
        self.trigger = trigger
        self.word = word
        self.trigger_begin_index = beginIndex
        self.trigger_end_index = endIndex

    def fitArgument(self,words,tags):
        """传入分词和标签列表，从中选择参数"""
        raise NotImplementedError()


    def __findFoward__ (self,words,tags,target,quickStop=False):
        '''find from 0 to self.trigger_begin_index'''
        '''if quickStop = True ,return when first find target, quickStop=False, return until find last target'''
        targetWord = ''
        # 从头找到触发词index
        for index in range(0, self.trigger_begin_index):
            if (tags[index] == 'B_'+target):  # 先找B_ ,后面找到的覆盖前面找到的
                targetWord = words[index]
            elif (tags[index] == 'I_'+target):  # 再找I_
                targetWord += words[index]
        return targetWord
    def __findBack__ (self,words,tags,target):
        targetWord = ''
        end_index = self.trigger_end_index+1
        # 从后往前找，直到触发词index
        hasFound = False
        while(end_index<len(words)):
            if (tags[end_index] == 'B_'+target):  # 先找B_
                if(hasFound):#找最近的，如果找到了就直接结束
                    break
                targetWord = words[end_index]
                hasFound = True
            elif (tags[end_index] == 'I_'+target):  # 再找I_
                targetWord += words[end_index]
            else:
                if(hasFound):
                    break
            end_index += 1 ;

        return targetWord


#相识事实有一个时间参数，往前找
class Know(baseModel):
    def fitArgument(self,words,tags):
        self.time = self.__findFoward__(words,tags,'Time')

    def __str__(self):
        if(self.time!=None and len(self.time)>0):
            return  self.time+ ' ' +self.word
        else:
            return self.word
class BeInLove(baseModel):
    def fitArgument(self, words, tags):
        self.time = self.__findFoward__(words, tags,'Time')

    def __str__(self):
        if (self.time != None and len(self.time) > 0):
            return  self.time+ ' ' +self.word
        else:
            return self.word
class Marray(baseModel):
    def fitArgument(self, words, tags):
        self.time = self.__findFoward__(words, tags,'Time')

    def __str__(self):
        if (self.time != None and len(self.time) > 0):
            return  self.time+ ' ' +self.word
        else:
            return self.word
class Remarray(baseModel):
    def fitArgument(self, words, tags):
        self.person = self.__findFoward__(words, tags,'Person')

    def __str__(self):
        if (self.person != None and len(self.person) > 0):
            return self.person+' '+self.word
        else:
            return self.word
class Bear(baseModel):
    def fitArgument(self,words,tags):
        self.time = self.__findFoward__(words, tags,'Time') #前面找时间
        #后面找其他参数
        self.gender = self.__findBack__(words,tags,'Gender')#后面找性别
        self.name = self.__findBack__(words,tags,'Name')#后面找姓名
        self.age = self.__findBack__(words,tags,'Age')#后面找年龄
    def __str__(self):
        str = self.word
        if(self.time!=None and len(self.time)>0):
            str = self.time+ ' ' +str
        if (self.gender != None and len(self.gender) > 0):
            str = str+' '+self.gender
        if (self.name != None and len(self.name) > 0):
            str = str + ' ' + self.name
        if (self.age != None and len(self.age) > 0):
            str = str+' '+self.age
        return str
class FamilyConflict(baseModel):
    def fitArgument(self,words,tags):
        return
    def __str__(self):
        return self.word

class DomesticViolence(baseModel):
    def fitArgument(self,words,tags):
        self.victim = self.__findBack__(words,tags,'Person')
        if(self.victim!=None and len(self.victim)>0):
            self.perpetrators = self.__findFoward__(words,tags,'Person')
        else:
            self.victim = self.__findFoward__(words,tags,'Person')
            self.perpetrators = self.__findFoward__(words,tags,'Person',quickStop=True)
            if(self.victim==self.perpetrators):
                self.perpetrators = None
                self.victim = None
    def __str__(self):
        str = self.word
        if(self.perpetrators!=None and len(self.perpetrators)>0):
            str = self.perpetrators +' ' + str
        if(self.victim!=None and len(self.victim)>0):
            str = str+' '+self.victim
        return str

class BadHabit(baseModel):
    def fitArgument(self,words,tags):
        self.person = self.__findFoward__(words,tags,'Person')
    def __str__(self):
        if(self.person!=None and len(self.person)>0):
            return self.person+' '+self.word
        else:
            return self.word

#出轨
class Derailed(baseModel):
    def fitArgument(self,words,tags):
        self.derailer = self.__findFoward__(words,tags,'Person')#出轨人
        self.time = self.__findFoward__(words,tags,'Time')#时间
    def __str__(self):
        str = ''
        if(self.time!=None and len(self.time)>0):
            str = self.time+' '+str
        if(self.derailer!=None and len(self.derailer)>0):
            str = self.derailer+' '+str
        str = str + self.word
        return str

class Separation(baseModel):
    def fitArgument(self,words,tags):
        self.beginTime = self.__findFoward__(words,tags,'Time')
        self.duration = self.__findFoward__(words,tags,'Duration')
    def __str__(self):
        str = self.word
        if (self.beginTime != None and len(self.beginTime) > 0):
            str = self.beginTime + ' ' + str
        if (self.duration != None and len(self.duration) > 0):
            str = str+' 持续：'+self.duration
        return str
class DivorceLawsuit(baseModel):
    def fitArgument(self,words,tags):
        self.beginTime = self.__findFoward__(words,tags,'Time')
        self.person = self.__findFoward__(words,tags,'Person')
        self.court = self.__findBack__(words,tags,'Court')
        if(self.court==None or len(self.court)==0):
            self.court = self.__findFoward__(words,tags,'Court')
        self.endTime = self.__findBack__(words,tags,'Time')
        self.document = self.__findBack__(words,tags,'Document')
        self.result = self.__findBack__(words,tags,'Judgment')
    def __str__(self):
        str = ''
        if (self.beginTime != None and len(self.beginTime) > 0):
            str = str+self.beginTime
        if (self.person != None and len(self.person) > 0):
            if(len(str)>0):
                str = str+' '+self.person
            else:
                str = self.person
        if(len(str)>0):
            str = str+' '+self.word
        else:
            str = self.word
        if (self.endTime != None and len(self.endTime) > 0):
            str = str+' '+self.endTime
        if (self.court != None and len(self.court) > 0):
            str = str+' '+self.court
        if (self.document != None and len(self.document) > 0):
            str = str+' '+self.document
        if (self.result != None and len(self.result) > 0):
            str = str+' '+self.result
        return str

class Wealth(baseModel):
    def fitArgument(self,words,tags):
        return
    def __str__(self):
        return self.word
class Debt(baseModel):
    def fitArgument(self,words,tags):
        return
    def __str__(self):
        return self.word
class Credit(baseModel):
    def fitArgument(self,words,tags):
        return
    def __str__(self):
        return self.word
if __name__ == '__main__':
    sys.exit(0)