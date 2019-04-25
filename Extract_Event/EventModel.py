#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

from Config.config_parser import getParser
__doc__ = 'description'
__author__ = '13314409603@163.com'



def EventFactory(type,completeWord,tag_index_pair,sentence,index_pairs,words,tags):
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
    return eventDict[type](type,completeWord,tag_index_pair,sentence,index_pairs,words,tags)
#基于规则匹配每类事实参数
class baseModel(object):
    def __init__(self,type,completeWord,tag_index_pair,sentence,sentence_char_index_pair,words,tags):
        self.type = type
        self.trigger = completeWord
        self.tag_index_pair = tag_index_pair
        self.trigger_index_pair = sentence_char_index_pair
        self.sentence = sentence
        self.negated,self.negated_index_pair = self.__findAll__(words,tags,'Negated',sentence_char_index_pair)

    def fitArgument(self,words,tags,words_in_sentence_index_pair):
        """传入分词和标签列表，从中选择参数"""
        raise NotImplementedError()


    def __findFoward__ (self,words,tags,target,words_in_sentence_index_pair,quickStop=False):
        '''find from 0 to self.trigger_begin_index'''
        '''if quickStop = True ,return when first find target, quickStop=False, return until find last target'''
        targetWord = ''
        index_pair = [0,0]
        haveFound = False
        # 从头找到触发词index
        for index in range(0, self.tag_index_pair[0]):
            if (tags[index] == 'B_'+target):  # 先找B_ ,后面找到的覆盖前面找到的
                if(haveFound and quickStop):
                    break
                targetWord = words[index]
                index_pair[0] = words_in_sentence_index_pair[index][0]
                index_pair[1] = words_in_sentence_index_pair[index][1]
                haveFound=True
            elif (tags[index] == 'I_'+target):  # 再找I_
                targetWord += words[index]
                index_pair[1] = words_in_sentence_index_pair[index][1]
        if(len(targetWord)==0):
            index_pair = None
        return targetWord,index_pair
    def __findBack__ (self,words,tags,target,words_in_sentence_index_pair):
        targetWord = ''
        index_pair = [0,0]
        end_index = self.tag_index_pair[1]+1
        # 从触发词开始往后找
        hasFound = False
        while(end_index<len(words)):
            if (tags[end_index] == 'B_'+target):  # 先找B_
                if(hasFound):#找最近的，如果找到了就直接结束
                    break
                targetWord = words[end_index]
                index_pair[0] = words_in_sentence_index_pair[end_index][0]
                index_pair[1] = words_in_sentence_index_pair[end_index][1]
                hasFound = True
            elif (tags[end_index] == 'I_'+target):  # 再找I_
                targetWord += words[end_index]
                index_pair[1] = words_in_sentence_index_pair[end_index][1]
            else:
                if(hasFound):
                    break
            end_index += 1
        if (len(targetWord) == 0):
            index_pair = None
        return targetWord,index_pair
    def __findAll__(self,words,tags,target,words_in_sentence_index_pair):
        #先往前找
        targetWord, index_pair = self.__findFoward__(words,tags,target,words_in_sentence_index_pair)
        if(not targetWord or len(targetWord)==0):
            targetWord,index_pair = self.__findBack__(words,tags,target,words_in_sentence_index_pair)
        return targetWord,index_pair

#相识事实有一个时间参数，往前找
class Know(baseModel):

    def fitArgument(self,words,tags,words_in_sentence_index_pair):
        '''匹配规则：在触发词前的Time参数'''
        self.time,self.time_index_pair = self.__findFoward__(words,tags,'Time',words_in_sentence_index_pair)


class BeInLove(baseModel):
    def fitArgument(self, words, tags,words_in_sentence_index_pair):
        '''匹配规则：触发词前的Time参数'''
        self.time,self.time_index_pair = self.__findFoward__(words, tags,'Time',words_in_sentence_index_pair)

class Marray(baseModel):
    def fitArgument(self, words, tags,words_in_sentence_index_pair):
        '''触发词前的第一个Time参数'''
        self.time,self.time_index_pair = self.__findFoward__(words, tags,'Time',words_in_sentence_index_pair)

class Remarray(baseModel):
    def fitArgument(self, words, tags,words_in_sentence_index_pair):
        self.person,self.person_index_pair = self.__findFoward__(words, tags,'Person',words_in_sentence_index_pair)

class Bear(baseModel):
    def fitArgument(self,words,tags,words_in_sentence_index_pair):
        '''匹配规则：在触发词前第一个time参数为出生时间；触发词后找gender,childeName,childAge'''
        self.dateOfBirth,self.dateOfBirth_index_pair = self.__findFoward__(words, tags,'Time',words_in_sentence_index_pair) #前面找时间
        #全句找其他参数
        self.gender,self.gender_index_pair = self.__findBack__(words,tags,'Gender',words_in_sentence_index_pair)
        self.childName,self.childName_index_pair = self.__findBack__(words,tags,'Name',words_in_sentence_index_pair)
        self.childAge,self.childAge_index_pair = self.__findBack__(words,tags,'Age',words_in_sentence_index_pair)
class FamilyConflict(baseModel):
    def fitArgument(self,words,tags,words_in_sentence_index_pair):
        return

class DomesticViolence(baseModel):
    def fitArgument(self,words,tags,words_in_sentence_index_pair):
        '''规则：1、 A 施暴 B；2、A 对 B 施暴；3、A 被 B 施暴'''

        #基于规则1
        self.victim,self.victim_index_pair = self.__findBack__(words,tags,'Person',words_in_sentence_index_pair)
        if(self.victim!=None and len(self.victim)>0):
            self.perpetrator,self.perpetrator_index_pair = self.__findFoward__(words,tags,'Person',words_in_sentence_index_pair)
        else:
            #基于规则2
            self.victim,self.victim_index_pair = self.__findFoward__(words,tags,'Person',words_in_sentence_index_pair)
            self.perpetrator,self.perpetrator_index_pair = self.__findFoward__(words,tags,'Person',words_in_sentence_index_pair,quickStop=True)
            if(self.victim==self.perpetrator):
                #如果只有一个对象，则无法确认
                self.perpetrator = ''
                self.perpetrator_index_pair = None
                self.victim = ''
                self.victim_index_pair = None
            else:
                #规则3
                if(self.sentence.find('被')!=-1):
                    word_tmp,index_pair_tmp = self.victim,self.victim_index_pair
                    self.victim,self.victim_index_pair = self.perpetrator,self.perpetrator_index_pair
                    self.perpetrator,self.perpetrator_index_pair = word_tmp,index_pair_tmp


class BadHabit(baseModel):
    def fitArgument(self,words,tags,words_in_sentence_index_pair):
        '''在触发词前找当事人'''
        self.person,self.person_index_pair = self.__findFoward__(words,tags,'Person',words_in_sentence_index_pair)

#出轨
class Derailed(baseModel):
    def fitArgument(self,words,tags,words_in_sentence_index_pair):
        '''在触发前找出轨人和时间'''
        self.derailer,self.derailer_index_pair = self.__findFoward__(words,tags,'Person',words_in_sentence_index_pair)#出轨人
        self.time,self.time_index_pair = self.__findFoward__(words,tags,'Time',words_in_sentence_index_pair)#时间

class Separation(baseModel):
    def fitArgument(self,words,tags,words_in_sentence_index_pair):
        '''在触发词前找开始时间，在触发词后找结束时间，在全句范围找持续时间'''
        self.beginTime,self.beginTime_index_pair = self.__findFoward__(words,tags,'Time',words_in_sentence_index_pair)
        self.endTime,self.endTime_index_pair = self.__findBack__(words,tags,'Time',words_in_sentence_index_pair)
        self.duration,self.duration_index_pair = self.__findAll__(words,tags,'Duration',words_in_sentence_index_pair)

class DivorceLawsuit(baseModel):
    def fitArgument(self,words,tags,words_in_sentence_index_pair):
        '''在触发词前找起诉人和起诉时间，在触发词后找判决时间；在全句范围找法院、判决书、判决结果'''
        self.sueTime,self.sueTime_index_pair = self.__findFoward__(words,tags,'Time',words_in_sentence_index_pair)
        self.initiator,self.initiator_index_pair = self.__findFoward__(words,tags,'Person',words_in_sentence_index_pair)
        self.court,self.court_index_pair = self.__findAll__(words,tags,'Court',words_in_sentence_index_pair)
        self.judgeTime,self.judgeTime_index_pair = self.__findBack__(words,tags,'Time',words_in_sentence_index_pair)
        self.judgeDocument,self.judgeDocument_index_pair = self.__findAll__(words,tags,'Document',words_in_sentence_index_pair)
        self.result,self.result_index_pair = self.__findAll__(words,tags,'Judgment',words_in_sentence_index_pair)

class Wealth(baseModel):
    def fitArgument(self,words,tags,words_in_sentence_index_pair):
        '''在全句范围找value,isCommon,isPersonal,whose'''
        self.value,self.value_index_pair = self.__findAll__(words,tags,'Price',words_in_sentence_index_pair)
        self.isCommon,self.isCommon_index_pair = self.__findAll__(words,tags,'CommonProperty',words_in_sentence_index_pair)
        self.isPersonal,self.isPersonal_index_pair = self.__findAll__(words,tags,'PersonalProperty',words_in_sentence_index_pair)
        self.whose,self.whose_index_pair = self.__findAll__(words,tags,'Person',words_in_sentence_index_pair)
        return
class Debt(baseModel):
    def fitArgument(self,words,tags,words_in_sentence_index_pair):
        '''在触发词前找债权人，在触发词后找价值'''
        self.creditor,self.creditor_index_pair = self.__findFoward__(words,tags,'Person',words_in_sentence_index_pair)
        self.value,self.value_index_pair = self.__findBack__(words,tags,'Price',words_in_sentence_index_pair)

        return
class Credit(baseModel):
    def fitArgument(self,words,tags,words_in_sentence_index_pair):
        '''在触发词前找债务人，在触发词后找价值'''
        self.creditor, self.creditor_index_pair = self.__findFoward__(words, tags, 'Person',
                                                                      words_in_sentence_index_pair)
        self.value, self.value_index_pair = self.__findBack__(words, tags, 'Price', words_in_sentence_index_pair)
        return

if __name__ == '__main__':
    sys.exit(0)