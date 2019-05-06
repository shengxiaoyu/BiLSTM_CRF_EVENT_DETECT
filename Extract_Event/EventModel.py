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

def EventFactory2(words,tags):
    eventDict = {
        'Know': Know,
        'BeInLove': BeInLove,
        'Marry': Marray,
        'Remarry': Remarray,
        'Bear': Bear,
        'FamilyConflict': FamilyConflict,
        'DomesticViolence': DomesticViolence,
        'BadHabit': BadHabit,
        'Derailed': Derailed,
        'Separation': Separation,
        'DivorceLawsuit': DivorceLawsuit,
        'Wealth': Wealth,
        'Debt': Debt,
        'Credit': Credit,
    }
    type = ''
    for tag in tags:
        if(tag.find('_Trigger')!=-1):
            type = tag[2:-8]
            break
    if(type in eventDict):
        event = eventDict[type].to_simple_model(type,words,tags)
        event.fit_arguments_by_spe(words,tags)
        return event
    else:
        return None

#基于规则匹配每类事实参数
class baseModel(object):
    def __init__(self,type,completeWord,tag_index_pair,sentence,sentence_char_index_pair,words,tags):
        self.type = type
        self.trigger = completeWord
        self.tag_index_pair = tag_index_pair
        self.trigger_index_pair = [sentence_char_index_pair[tag_index_pair[0]][0],sentence_char_index_pair[tag_index_pair[1]][1]]
        self.sentence = sentence
        self.negated,self.negated_index_pair = self.__findAll__(words,tags,'Negated',sentence_char_index_pair)

    def fitArgument(self,words,tags,words_in_sentence_index_pair):
        """传入分词和标签列表，以及每个分词再原句中的起止索引号，从中选择参数"""
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
        while(end_index<len(tags)):
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

    @classmethod
    def to_simple_model(cls,type, words, tags):
        trigger = ''
        tag_index_pair = [0, 0]
        for index, tag in enumerate(tags):
            if (tag.find('_Trigger') != -1):
                trigger = words[index]
                tag_index_pair[0]=index
                tag_index_pair[1]=index
                for theIndex in range(index + 1, len(tags)):
                    if (tags[theIndex].find('_Trigger') != -1):
                        trigger = trigger + words[theIndex]
                        tag_index_pair[1]=theIndex
                break
        sentence = ''
        sentence_char_index_pair = [[0, 0] for _ in range(len(tags))]
        return cls(type, trigger, tag_index_pair, sentence, sentence_char_index_pair, words, tags)

    def fit_arguments_by_spe(self, words, new_tags):
        """传入分词和标签列表，从中选择参数"""
        raise NotImplementedError()

    def get_score(self):
        raise NotImplementedError()
    def compare(self,another_baseModel):
        raise NotImplementedError()

#相识事实有一个时间参数，往前找
class Know(baseModel):
    def fitArgument(self,words,tags,words_in_sentence_index_pair):
        '''匹配规则：在触发词前的Time参数'''
        self.time,self.time_index_pair = self.__findFoward__(words,tags,'Time',words_in_sentence_index_pair)
    def fit_arguments_by_spe(self,words,new_tags):
        self.time = ''
        for index,tag in enumerate(new_tags):
            if(tag.find('B_Know_Time')!=-1):
                self.time = words[index]
                for theIndex in range(index+1,len(new_tags)):
                    if(new_tags[theIndex]=='I_Know_Time'):
                        self.time = self.time+words[theIndex]
                break

    def get_score(self):
        score = 1
        if (self.time != None and self.time != ''):
            score += 1
        return score

    def compare(self, another_baseModel):
        score = 0
        if (self.trigger == another_baseModel.trigger):
            score += 1
        if (self.time!=None and self.time!='' and self.time == another_baseModel.time):
            score += 1
        return score

class BeInLove(baseModel):
    def fitArgument(self, words, tags,words_in_sentence_index_pair):
        '''匹配规则：触发词前的Time参数'''
        self.time,self.time_index_pair = self.__findFoward__(words, tags,'Time',words_in_sentence_index_pair)
    def fit_arguments_by_spe(self,words,new_tags):
        self.time = ''
        for index,tag in enumerate(new_tags):
            if(tag.find('B_BeInLove_Time')!=-1):
                self.time = words[index]
                for theIndex in range(index+1,len(new_tags)):
                    if(new_tags[theIndex]=='I_BeInLove_Time'):
                        self.time = self.time+words[theIndex]
                break

    def get_score(self):
        score = 1
        if(self.time!=None and self.time!=''):
            score+=1
        return score
    def compare(self,another_baseModel):
        score = 0
        if(self.trigger==another_baseModel.trigger):
            score+=1
        if(self.time!=None and self.time!='' and self.time==another_baseModel.time):
            score +=1
        return score
class Marray(baseModel):
    def fitArgument(self, words, tags,words_in_sentence_index_pair):
        '''触发词前的第一个Time参数'''
        self.time,self.time_index_pair = self.__findFoward__(words, tags,'Time',words_in_sentence_index_pair)
    def fit_arguments_by_spe(self,words,new_tags):
        self.time = ''
        for index,tag in enumerate(new_tags):
            if(tag.find('B_Marry_Time')!=-1):
                self.time = words[index]
                for theIndex in range(index+1,len(new_tags)):
                    if(new_tags[theIndex]=='I_Marry_Time'):
                        self.time = self.time+words[theIndex]
                break

    def get_score(self):
        score = 1
        if (self.time != None and self.time != ''):
            score += 1
        return score

    def compare(self, another_baseModel):
        score = 0
        if (self.trigger == another_baseModel.trigger):
            score += 1
        if (self.time!=None and self.time!='' and self.time == another_baseModel.time):
            score += 1
        return score
class Remarray(baseModel):
    def fitArgument(self, words, tags,words_in_sentence_index_pair):
        self.person,self.person_index_pair = self.__findFoward__(words, tags,'Person',words_in_sentence_index_pair)

    def fit_arguments_by_spe(self,words,new_tags):
        self.person = ''
        for index,tag in enumerate(new_tags):
            if(tag.find('B_Remarry_Participant')!=-1):
                self.person = words[index]
                for theIndex in range(index+1,len(new_tags)):
                    if(new_tags[theIndex]=='I_Remarry_Participant'):
                        self.person = self.person+words[theIndex]
                break

    def get_score(self):
        score = 1
        if (self.person != None and self.person != ''):
            score += 1
        return score

    def compare(self, another_baseModel):
        score = 0
        if (self.trigger == another_baseModel.trigger):
            score += 1
        if (self.person != None and self.person != '' and self.person == another_baseModel.person):
            score += 1
        return score
class Bear(baseModel):
    def fitArgument(self,words,tags,words_in_sentence_index_pair):
        '''匹配规则：在触发词前第一个time参数为出生时间；触发词后找gender,childeName,childAge'''
        self.dateOfBirth,self.dateOfBirth_index_pair = self.__findFoward__(words, tags,'Time',words_in_sentence_index_pair) #前面找时间
        #全句找其他参数
        self.gender,self.gender_index_pair = self.__findBack__(words,tags,'Gender',words_in_sentence_index_pair)
        self.childName,self.childName_index_pair = self.__findBack__(words,tags,'Name',words_in_sentence_index_pair)
        self.childAge,self.childAge_index_pair = self.__findBack__(words,tags,'Age',words_in_sentence_index_pair)
    def fit_arguments_by_spe(self,words,new_tags):
        self.dateOfBirth = ''
        self.gender = ''
        self.childName =''
        self.childAge = ''
        for index,tag in enumerate(new_tags):
            if(tag.find('B_Bear_DateOfBirth')!=-1):
                self.dateOfBirth = words[index]
                for theIndex in range(index+1,len(new_tags)):
                    if(new_tags[theIndex]=='I_Bear_DateOfBirth'):
                        self.dateOfBirth = self.dateOfBirth+words[theIndex]
            if (tag.find('B_Bear_Gender') != -1):
                self.gender = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_Bear_Gender'):
                        self.gender = self.gender + words[theIndex]
            if (tag.find('B_Bear_ChildName') != -1):
                self.childName = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_Bear_ChildName'):
                        self.childName = self.childName + words[theIndex]
            if (tag.find('B_Bear_Age') != -1):
                self.childAge = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_Bear_Age'):
                        self.childAge = self.childAge + words[theIndex]
    def get_score(self):
        score = 1
        if(self.dateOfBirth!=None and self.dateOfBirth!=''):
            score+=1
        if (self.gender != None and self.gender != ''):
            score += 1
        if (self.childName != None and self.childName != ''):
            score += 1
        if (self.childAge != None and self.childAge != ''):
            score += 1
        return score
    def compare(self, another_baseModel):
        score = 0
        if (self.trigger == another_baseModel.trigger):
            score += 1
        if (self.dateOfBirth!=None and self.dateOfBirth!='' and self.dateOfBirth!=''and self.dateOfBirth!=None and self.dateOfBirth == another_baseModel.dateOfBirth):
            score += 1
        if (self.gender != None and self.gender != '' and self.gender!=''and self.gender!=None and self.gender == another_baseModel.gender):
            score += 1
        if (self.childAge != None and self.childAge != '' and self.childAge!=''and self.childAge!=None and self.childAge == another_baseModel.childAge):
            score += 1
        if (self.childName != None and self.childName != '' and self.childName!=''and self.childName!=None and self.childName == another_baseModel.childName):
            score += 1
        return score
class FamilyConflict(baseModel):
    def fitArgument(self,words,tags,words_in_sentence_index_pair):
        return

    def fit_arguments_by_spe(self, words, new_tags):
        return

    def get_score(self):
        score = 1
        return score
    def compare(self, another_baseModel):
        score = 0
        if (self.trigger == another_baseModel.trigger):
            score += 1

        return score
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

    def fit_arguments_by_spe(self,words,new_tags):
        self.victim = ''
        self.perpetrator = ''
        for index,tag in enumerate(new_tags):
            if(tag.find('B_DomesticViolence_Victim')!=-1):
                self.victim = words[index]
                for theIndex in range(index+1,len(new_tags)):
                    if(new_tags[theIndex]=='I_DomesticViolence_Victim'):
                        self.victim = self.victim+words[theIndex]
            if (tag.find('B_DomesticViolence_Perpetrators') != -1):
                self.perpetrator = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_DomesticViolence_Perpetrators'):
                        self.perpetrator = self.perpetrator + words[theIndex]
    def get_score(self):
        score = 1
        if(self.victim !=None and self.victim!='' ):
            score+=1
        if (self.perpetrator!=None and self.perpetrator!=''):
            score += 1
        return score
    def compare(self, another_baseModel):
        score = 0
        if (self.trigger == another_baseModel.trigger):
            score += 1
        if (self.victim !=None and self.victim!='' and self.victim == another_baseModel.victim):
            score += 1
        if (self.perpetrator!=None and self.perpetrator!='' and self.perpetrator == another_baseModel.perpetrator):
            score += 1
        return score
class BadHabit(baseModel):
    def fitArgument(self,words,tags,words_in_sentence_index_pair):
        '''在触发词前找当事人'''
        self.person,self.person_index_pair = self.__findFoward__(words,tags,'Person',words_in_sentence_index_pair)
    def fit_arguments_by_spe(self,words,new_tags):
        self.person = ''
        for index,tag in enumerate(new_tags):
            if(tag.find('B_BadHabit_Participant')!=-1):
                self.person = words[index]
                for theIndex in range(index+1,len(new_tags)):
                    if(new_tags[theIndex]=='I_BadHabit_Participant'):
                        self.person = self.person+words[theIndex]
                break
    def get_score(self):
        score = 1
        if(self.person !=None and self.person!='' ):
            score+=1
        return score
    def compare(self, another_baseModel):
        score = 0
        if (self.trigger == another_baseModel.trigger):
            score += 1
        if (self.person !=None and self.person!='' and self.person == another_baseModel.person):
            score += 1
        return score
#出轨
class Derailed(baseModel):
    def fitArgument(self,words,tags,words_in_sentence_index_pair):
        '''在触发前找出轨人和时间'''
        self.derailer,self.derailer_index_pair = self.__findFoward__(words,tags,'Person',words_in_sentence_index_pair)#出轨人
        self.time,self.time_index_pair = self.__findFoward__(words,tags,'Time',words_in_sentence_index_pair)#时间
    def fit_arguments_by_spe(self,words,new_tags):
        self.derailer = ''
        self.time = ''
        for index,tag in enumerate(new_tags):
            if(tag.find('B_Derailed_Derailer')!=-1):
                self.derailer = words[index]
                for theIndex in range(index+1,len(new_tags)):
                    if(new_tags[theIndex]=='I_Derailed_Derailer'):
                        self.derailer = self.derailer+words[theIndex]
            if (tag.find('B_Derailed_Time') != -1):
                self.time = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_Derailed_Time'):
                        self.time = self.time + words[theIndex]

    def get_score(self):
        score = 1
        if(self.derailer !=None and self.derailer!='' ):
            score+=1
        if (self.time != None and self.time != ''):
            score += 1
        return score
    def compare(self, another_baseModel):
        score = 0
        if (self.trigger == another_baseModel.trigger):
            score += 1
        if (self.derailer !=None and self.derailer!='' and self.derailer == another_baseModel.derailer):
            score += 1
        if (self.time != None and self.time != '' and self.time == another_baseModel.time):
            score += 1
        return score
class Separation(baseModel):
    def fitArgument(self,words,tags,words_in_sentence_index_pair):
        '''在触发词前找开始时间，在触发词后找结束时间，在全句范围找持续时间'''
        self.beginTime,self.beginTime_index_pair = self.__findFoward__(words,tags,'Time',words_in_sentence_index_pair)
        self.endTime,self.endTime_index_pair = self.__findBack__(words,tags,'Time',words_in_sentence_index_pair)
        self.duration,self.duration_index_pair = self.__findAll__(words,tags,'Duration',words_in_sentence_index_pair)
    def fit_arguments_by_spe(self,words,new_tags):
        self.beginTime = ''
        self.endTime = ''
        self.duration = ''
        for index,tag in enumerate(new_tags):
            if(tag.find('B_Separation_BeginTime')!=-1):
                self.beginTime = words[index]
                for theIndex in range(index+1,len(new_tags)):
                    if(new_tags[theIndex]=='I_Separation_BeginTime'):
                        self.beginTime = self.beginTime+words[theIndex]
            if (tag.find('B_Separation_EndTime') != -1):
                self.endTime = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_Separation_EndTime'):
                        self.endTime = self.endTime + words[theIndex]
            if (tag.find('B_Separation_Duration') != -1):
                self.duration = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_Separation_Duration'):
                        self.duration = self.duration + words[theIndex]
    def get_score(self):
        score = 1
        if(self.beginTime !=None and self.beginTime!='' ):
            score+=1
        if (self.endTime != None and self.endTime != ''):
            score += 1
        if (self.duration != None and self.duration != ''):
            score += 1
        return score
    def compare(self, another_baseModel):
        score = 0
        if (self.trigger == another_baseModel.trigger):
            score += 1
        if (self.beginTime !=None and self.beginTime!=''  and self.beginTime == another_baseModel.beginTime):
            score += 1
        if (self.endTime != None and self.endTime != ''and self.endTime == another_baseModel.endTime):
            score += 1
        if (self.duration != None and self.duration != '' and self.duration == another_baseModel.duration):
            score += 1
        return score
class DivorceLawsuit(baseModel):
    def fitArgument(self,words,tags,words_in_sentence_index_pair):
        '''在触发词前找起诉人和起诉时间，在触发词后找判决时间；在全句范围找法院、判决书、判决结果'''
        self.sueTime,self.sueTime_index_pair = self.__findFoward__(words,tags,'Time',words_in_sentence_index_pair)
        self.initiator,self.initiator_index_pair = self.__findFoward__(words,tags,'Person',words_in_sentence_index_pair)
        self.court,self.court_index_pair = self.__findAll__(words,tags,'Court',words_in_sentence_index_pair)
        self.judgeTime,self.judgeTime_index_pair = self.__findBack__(words,tags,'Time',words_in_sentence_index_pair)
        self.judgeDocument,self.judgeDocument_index_pair = self.__findAll__(words,tags,'Document',words_in_sentence_index_pair)
        self.result,self.result_index_pair = self.__findAll__(words,tags,'Judgment',words_in_sentence_index_pair)
    def fit_arguments_by_spe(self,words,new_tags):
        self.sueTime = ''
        self.initiator = ''
        self.court =''
        self.judgeTime = ''
        self.judgeDocument = ''
        self.result = ''
        for index,tag in enumerate(new_tags):
            if(tag.find('B_DivorceLawsuit_SueTime')!=-1):
                self.sueTime = words[index]
                for theIndex in range(index+1,len(new_tags)):
                    if(new_tags[theIndex]=='I_DivorceLawsuit_SueTime'):
                        self.sueTime = self.sueTime+words[theIndex]
            if (tag.find('B_DivorceLawsuit_Initiator') != -1):
                self.initiator = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_DivorceLawsuit_Initiator'):
                        self.initiator = self.initiator + words[theIndex]
            if (tag.find('B_DivorceLawsuit_Court') != -1):
                self.court = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_DivorceLawsuit_Court'):
                        self.court = self.court + words[theIndex]
            if (tag.find('B_DivorceLawsuit_JudgeTime') != -1):
                self.judgeTime = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_DivorceLawsuit_JudgeTime'):
                        self.judgeTime = self.judgeTime + words[theIndex]
            if (tag.find('B_DivorceLawsuit_JudgeDocument') != -1):
                self.judgeDocument = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_DivorceLawsuit_JudgeDocument'):
                        self.judgeDocument = self.judgeDocument + words[theIndex]
            if (tag.find('B_DivorceLawsuit_Result') != -1):
                self.result = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_DivorceLawsuit_Result'):
                        self.result = self.result + words[theIndex]
    def get_score(self):
        score = 1
        if(self.sueTime !=None and self.sueTime!='' ):
            score+=1
        if (self.initiator != None and self.initiator != ''):
            score += 1
        if (self.court != None and self.court != ''):
            score += 1
        if (self.result != None and self.result != ''):
            score += 1
        if (self.judgeTime != None and self.judgeTime != ''):
            score += 1
        if (self.judgeDocument != None and self.judgeDocument != ''):
            score += 1
        return score
    def compare(self, another_baseModel):
        score = 0
        if (self.trigger == another_baseModel.trigger):
            score += 1
        if (self.sueTime !=None and self.sueTime!='' and self.sueTime == another_baseModel.sueTime):
            score += 1
        if (self.initiator != None and self.initiator != ''and self.initiator == another_baseModel.initiator):
            score += 1
        if (self.court != None and self.court != ''and self.court == another_baseModel.court):
            score += 1
        if (self.result != None and self.result != ''and self.result == another_baseModel.result):
            score += 1
        if (self.judgeTime != None and self.judgeTime != '' and self.judgeTime == another_baseModel.judgeTime):
            score += 1
        if (self.judgeDocument != None and self.judgeDocument != '' and self.judgeDocument == another_baseModel.judgeDocument):
            score += 1
        return score
class Wealth(baseModel):
    def fitArgument(self,words,tags,words_in_sentence_index_pair):
        '''在全句范围找value,isCommon,isPersonal,whose'''
        self.value,self.value_index_pair = self.__findAll__(words,tags,'Price',words_in_sentence_index_pair)
        self.isCommon,self.isCommon_index_pair = self.__findAll__(words,tags,'CommonProperty',words_in_sentence_index_pair)
        self.isPersonal,self.isPersonal_index_pair = self.__findAll__(words,tags,'PersonalProperty',words_in_sentence_index_pair)
        self.whose,self.whose_index_pair = self.__findAll__(words,tags,'Person',words_in_sentence_index_pair)
        return
    def fit_arguments_by_spe(self,words,new_tags):
        self.value = ''
        self.isCommon = ''
        self.isPersonal =''
        self.whose = ''
        for index,tag in enumerate(new_tags):
            if(tag.find('B_Wealth_Value')!=-1):
                self.value = words[index]
                for theIndex in range(index+1,len(new_tags)):
                    if(new_tags[theIndex]=='I_Wealth_Value'):
                        self.value = self.value+words[theIndex]
            if (tag.find('B_Wealth_IsCommon') != -1):
                self.isCommon = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_Wealth_IsCommon'):
                        self.isCommon = self.isCommon + words[theIndex]
            if (tag.find('B_Wealth_IsPersonal') != -1):
                self.isPersonal = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_Wealth_IsPersonal'):
                        self.isPersonal = self.isPersonal + words[theIndex]
            if (tag.find('B_Wealth_Whose') != -1):
                self.whose = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_Wealth_Whose'):
                        self.whose = self.whose + words[theIndex]
    def get_score(self):
        score = 1
        if(self.value !=None and self.value!='' ):
            score+=1
        if (self.isCommon != None and self.isCommon != ''):
            score += 1
        if (self.isPersonal != None and self.isPersonal != ''):
            score += 1
        if (self.whose != None and self.whose != ''):
            score += 1
        return score
    def compare(self, another_baseModel):
        score = 0
        if (self.trigger == another_baseModel.trigger):
            score += 1
        if (self.value !=None and self.value!=''  and self.value == another_baseModel.value):
            score += 1
        if (self.isCommon != None and self.isCommon != '' and self.isCommon == another_baseModel.isCommon):
            score += 1
        if (self.isPersonal != None and self.isPersonal != '' and self.isPersonal == another_baseModel.isPersonal):
            score += 1
        if (self.whose != None and self.whose != '' and self.whose == another_baseModel.whose):
            score += 1
        return score
class Debt(baseModel):
    def fitArgument(self,words,tags,words_in_sentence_index_pair):
        '''在触发词前找债权人，在触发词后找价值'''
        self.creditor,self.creditor_index_pair = self.__findFoward__(words,tags,'Person',words_in_sentence_index_pair)
        self.value,self.value_index_pair = self.__findBack__(words,tags,'Price',words_in_sentence_index_pair)
        return

    def fit_arguments_by_spe(self,words,new_tags):
        self.value = ''
        self.creditor = ''
        for index,tag in enumerate(new_tags):
            if(tag.find('B_Debt_Value')!=-1):
                self.value = words[index]
                for theIndex in range(index+1,len(new_tags)):
                    if(new_tags[theIndex]=='I_Debt_Value'):
                        self.value = self.value+words[theIndex]
            if (tag.find('B_Debt_Creditor') != -1):
                self.creditor = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_Debt_Creditor'):
                        self.creditor = self.creditor + words[theIndex]
    def get_score(self):
        score = 1
        if(self.value !=None and self.value!='' ):
            score+=1
        if (self.creditor != None and self.creditor != ''):
            score += 1
        return score
    def compare(self, another_baseModel):
        score = 0
        if (self.trigger == another_baseModel.trigger):
            score += 1
        if (self.value !=None and self.value!=''  and self.value == another_baseModel.value):
            score += 1
        if (self.creditor != None and self.creditor != '' and self.creditor == another_baseModel.creditor):
            score += 1
        return score
class Credit(baseModel):
    def fitArgument(self,words,tags,words_in_sentence_index_pair):
        '''在触发词前找债务人，在触发词后找价值'''
        self.debtor, self.debtor_index_pair = self.__findFoward__(words, tags, 'Person',
                                                                      words_in_sentence_index_pair)
        self.value, self.value_index_pair = self.__findBack__(words, tags, 'Price', words_in_sentence_index_pair)
        return

    def fit_arguments_by_spe(self,words,new_tags):
        self.value = ''
        self.debtor = ''
        for index,tag in enumerate(new_tags):
            if(tag.find('B_Credit_Debtor')!=-1):
                self.debtor = words[index]
                for theIndex in range(index+1,len(new_tags)):
                    if(new_tags[theIndex]=='I_Credit_Debtor'):
                        self.debtor = self.debtor+words[theIndex]
            if (tag.find('B_Credit_Value') != -1):
                self.value = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_Credit_Value'):
                        self.value = self.value + words[theIndex]
    def get_score(self):
        score = 1
        if(self.value !=None and self.value!='' ):
            score+=1
        if (self.debtor != None and self.debtor != ''):
            score += 1
        return score
    def compare(self, another_baseModel):
        score = 0
        if (self.trigger == another_baseModel.trigger):
            score += 1
        if (self.value !=None and self.value!='' and self.value == another_baseModel.value):
            score += 1
        if (self.debtor != None and self.debtor != '' and self.debtor == another_baseModel.debtor):
            score += 1
        return score
if __name__ == '__main__':
    sys.exit(0)