#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

__doc__ = 'description'
__author__ = '13314409603@163.com'

def EventFactory(event_argu_dict,event_argus_index_pair_dict,sentence):
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
    event =  eventDict[event_argu_dict['Type']](event_argu_dict,event_argus_index_pair_dict,sentence)
    return event

#单句单事件构造,准确地基于标签类型
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

class baseModel(object):
    def __init__(self,argu_dict,event_argus_index_pair_dict,sentence):
        self.trigger = argu_dict['Trigger']
        self.type = argu_dict['Type']
        self.trigger_index_pair = event_argus_index_pair_dict['Trigger']
        self.sentence = sentence
        self.negated = argu_dict['Negated'] if('Negated' in argu_dict) else ""
        self.negated_index_pair = event_argus_index_pair_dict['Negated'] if('Negated' in event_argus_index_pair_dict) else None

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
            end_index += 1

        return targetWord

    def fit_arguments_by_spe(self, words, new_tags):
        """传入分词和标签列表，从中选择参数"""
        raise NotImplementedError()

    def get_score(self):
        raise NotImplementedError()

#相识事实有一个时间参数，往前找
class Know(baseModel):
    def __init__(self,argu_dict,event_argus_index_pair_dict,sentence):
        baseModel.__init__(self,argu_dict,event_argus_index_pair_dict,sentence)
        self.time =argu_dict['Know_Time'] if('Know_Time' in argu_dict) else ""
        self.time_index_pair =event_argus_index_pair_dict['Know_Time'] if('Know_Time' in event_argus_index_pair_dict) else None

    def __str__(self):
        str = self.negated+' '+self.time+' '+self.trigger
        return str.strip()

    def fit_arguments_by_spe(self, words, new_tags):
        self.time = ''
        for index, tag in enumerate(new_tags):
            if (tag.find('B_Know_Time') != -1):
                self.time = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_Know_Time'):
                        self.time = self.time + words[theIndex]
                break

    def get_score(self):
        score = 1
        if (self.time != None and self.time != ''):
            score += 1
        return score

class BeInLove(baseModel):
    def __init__(self, argu_dict, event_argus_index_pair_dict, sentence):
        baseModel.__init__(self, argu_dict, event_argus_index_pair_dict, sentence)
        self.time = argu_dict['BeInLove_Time'] if ('BeInLove_Time' in argu_dict) else ""
        self.time_index_pair = event_argus_index_pair_dict['BeInLove_Time'] if (
                    'BeInLove_Time' in event_argus_index_pair_dict) else None

    def __str__(self):
        str =  self.negated+' '+self.time + ' ' + self.trigger
        return str.strip()

    def fit_arguments_by_spe(self, words, new_tags):
        self.time = ''
        for index, tag in enumerate(new_tags):
            if (tag.find('B_BeInLove_Time') != -1):
                self.time = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_BeInLove_Time'):
                        self.time = self.time + words[theIndex]
                break

    def get_score(self):
        score = 1
        if (self.time != None and self.time != ''):
            score += 1
        return score

class Marray(baseModel):
    def __init__(self, argu_dict, event_argus_index_pair_dict, sentence):
        baseModel.__init__(self, argu_dict, event_argus_index_pair_dict, sentence)
        self.time = argu_dict['Marry_Time'] if ('Marry_Time' in argu_dict) else ""
        self.time_index_pair = event_argus_index_pair_dict['Marry_Time'] if (
                'Marry_Time' in event_argus_index_pair_dict) else None

    def __str__(self):
        str =  self.negated+' '+self.time + ' ' + self.trigger
        return str.strip()

    def fit_arguments_by_spe(self, words, new_tags):
        self.time = ''
        for index, tag in enumerate(new_tags):
            if (tag.find('B_Marry_Time') != -1):
                self.time = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_Marry_Time'):
                        self.time = self.time + words[theIndex]
                break

    def get_score(self):
        score = 1
        if (self.time != None and self.time != ''):
            score += 1
        return score


class Remarray(baseModel):
    def __init__(self, argu_dict, event_argus_index_pair_dict, sentence):
        baseModel.__init__(self, argu_dict, event_argus_index_pair_dict, sentence)
        self.participant = argu_dict['Remarry_Participant'] if ('Remarry_Participant' in argu_dict) else ""
        self.participant_index_pair = event_argus_index_pair_dict['Remarry_Participant'] if ('Remarry_Participant' in event_argus_index_pair_dict) else None

    def __str__(self):
        str =  self.negated+' '+self.participant+' '+self.trigger
        return str.strip()

    def fit_arguments_by_spe(self, words, new_tags):
        self.person = ''
        for index, tag in enumerate(new_tags):
            if (tag.find('B_Remarry_Participant') != -1):
                self.person = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_Remarry_Participant'):
                        self.person = self.person + words[theIndex]
                break

    def get_score(self):
        score = 1
        if (self.person != None and self.person != ''):
            score += 1
        return score

class Bear(baseModel):
    def __init__(self, argu_dict, event_argus_index_pair_dict, sentence):
        baseModel.__init__(self, argu_dict, event_argus_index_pair_dict, sentence)

        self.dateOfBirth = argu_dict['Bear_DateOfBirth'] if ('Bear_DateOfBirth' in argu_dict) else ""
        self.dateOfBirth_index_pair = event_argus_index_pair_dict['Bear_DateOfBirth'] if ('Bear_DateOfBirth' in event_argus_index_pair_dict) else None

        self.gender = argu_dict['Bear_Gender'] if ('Bear_Gender' in argu_dict) else ""
        self.gender_index_pair = event_argus_index_pair_dict['Bear_Gender'] if ('Bear_Gender' in event_argus_index_pair_dict) else None

        self.childName = argu_dict['Bear_ChildName'] if ('Bear_ChildName' in argu_dict) else ""
        self.childName_index_pair = event_argus_index_pair_dict['Bear_ChildName'] if (
                    'Bear_ChildName' in event_argus_index_pair_dict) else None

        self.age = argu_dict['Bear_Age'] if ('Bear_Age' in argu_dict) else ""
        self.age_index_pair = event_argus_index_pair_dict['Bear_Age'] if (
                    'Bear_Age' in event_argus_index_pair_dict) else None
    def __str__(self):
        str = self.negated+' '+((self.dateOfBirth+' ') if len(self.dateOfBirth)>0 else '')+self.trigger+' '+((self.gender+' ') if len(self.gender)>0 else '')\
              +((self.childName + ' ') if len(self.childName) > 0 else '') +((self.age + ' ') if len(self.age) > 0 else '')
        return str.strip()

    def fit_arguments_by_spe(self, words, new_tags):
        self.dateOfBirth = ''
        self.gender = ''
        self.childName = ''
        self.childAge = ''
        for index, tag in enumerate(new_tags):
            if (tag.find('B_Bear_DateOfBirth') != -1):
                self.dateOfBirth = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_Bear_DateOfBirth'):
                        self.dateOfBirth = self.dateOfBirth + words[theIndex]
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
        if (self.dateOfBirth != None and self.dateOfBirth != ''):
            score += 1
        if (self.gender != None and self.gender != ''):
            score += 1
        if (self.childName != None and self.childName != ''):
            score += 1
        if (self.childAge != None and self.childAge != ''):
            score += 1
        return score

class FamilyConflict(baseModel):
    def __init__(self, argu_dict, event_argus_index_pair_dict, sentence):
        baseModel.__init__(self, argu_dict, event_argus_index_pair_dict, sentence)
    def __str__(self):
        return self.trigger

    def fit_arguments_by_spe(self, words, new_tags):
        return

    def get_score(self):
        score = 1
        return score
class DomesticViolence(baseModel):
    def __init__(self, argu_dict, event_argus_index_pair_dict, sentence):
        baseModel.__init__(self, argu_dict, event_argus_index_pair_dict, sentence)

        self.time = argu_dict['DomesticViolence_Time'] if ('DomesticViolence_Time' in argu_dict) else ""
        self.time_index_pair = event_argus_index_pair_dict['DomesticViolence_Time'] if (
                'DomesticViolence_Time' in event_argus_index_pair_dict) else None

        self.perpetrator = argu_dict['DomesticViolence_Perpetrators'] if ('DomesticViolence_Perpetrators' in argu_dict) else ""
        self.perpetrator_index_pair = event_argus_index_pair_dict['DomesticViolence_Perpetrators'] if ('DomesticViolence_Perpetrators' in event_argus_index_pair_dict) else None

        self.victim = argu_dict['DomesticViolence_Victim'] if ('DomesticViolence_Victim' in argu_dict) else ""
        self.victim_index_pair = event_argus_index_pair_dict['DomesticViolence_Victim'] if ('DomesticViolence_Victim' in event_argus_index_pair_dict) else None


    def __str__(self):
        str = self.negated+' '+((self.time+' ') if len(self.time)>0 else '')+((self.perpetrator+' ') if len(self.perpetrator)>0 else '')+self.trigger+' '+((self.victim+' ') if len(self.victim)>0 else '')

        return str.strip()

    def fit_arguments_by_spe(self, words, new_tags):
        self.victim = ''
        self.perpetrator = ''
        for index, tag in enumerate(new_tags):
            if (tag.find('B_DomesticViolence_Victim') != -1):
                self.victim = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_DomesticViolence_Victim'):
                        self.victim = self.victim + words[theIndex]
            if (tag.find('B_DomesticViolence_Perpetrators') != -1):
                self.perpetrator = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_DomesticViolence_Perpetrators'):
                        self.perpetrator = self.perpetrator + words[theIndex]

    def get_score(self):
        score = 1
        if (self.victim != None and self.victim != ''):
            score += 1
        if (self.perpetrator != None and self.perpetrator != ''):
            score += 1
        return score
class BadHabit(baseModel):
    def __init__(self, argu_dict, event_argus_index_pair_dict, sentence):
        baseModel.__init__(self, argu_dict, event_argus_index_pair_dict, sentence)

        self.person = argu_dict['BadHabit_Participant'] if ('BadHabit_Participant' in argu_dict) else ""
        self.person_index_pair = event_argus_index_pair_dict['BadHabit_Participant'] if (
                'BadHabit_Participant' in event_argus_index_pair_dict) else None

    def __str__(self):
        str = self.negated+' '+((self.person+' ') if len(self.person)>0 else '')+self.trigger
        return str.strip()

    def fit_arguments_by_spe(self, words, new_tags):
        self.person = ''
        for index, tag in enumerate(new_tags):
            if (tag.find('B_BadHabit_Participant') != -1):
                self.person = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_BadHabit_Participant'):
                        self.person = self.person + words[theIndex]
                break

    def get_score(self):
        score = 1
        if (self.person != None and self.person != ''):
            score += 1
        return score
#出轨
class Derailed(baseModel):
    def __init__(self, argu_dict, event_argus_index_pair_dict, sentence):
        baseModel.__init__(self, argu_dict, event_argus_index_pair_dict, sentence)

        self.time = argu_dict['Derailed_Time'] if ('Derailed_Time' in argu_dict) else ""
        self.time_index_pair = event_argus_index_pair_dict['Derailed_Time'] if (
                'Derailed_Time' in event_argus_index_pair_dict) else None

        self.derailer = argu_dict['Derailed_Derailer'] if ('Derailed_Derailer' in argu_dict) else ""
        self.derailer_index_pair = event_argus_index_pair_dict['Derailed_Derailer'] if (
                'Derailed_Derailer' in event_argus_index_pair_dict) else None
    def __str__(self):
        str = self.negated+' '+((self.time+' ') if len(self.time)>0 else '')+((self.derailer+' ') if len(self.derailer)>0 else '')+self.trigger
        return str.strip()

    def fit_arguments_by_spe(self, words, new_tags):
        self.derailer = ''
        self.time = ''
        for index, tag in enumerate(new_tags):
            if (tag.find('B_Derailed_Derailer') != -1):
                self.derailer = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_Derailed_Derailer'):
                        self.derailer = self.derailer + words[theIndex]
            if (tag.find('B_Derailed_Time') != -1):
                self.time = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_Derailed_Time'):
                        self.time = self.time + words[theIndex]

    def get_score(self):
        score = 1
        if (self.derailer != None and self.derailer != ''):
            score += 1
        if (self.time != None and self.time != ''):
            score += 1
        return score

class Separation(baseModel):
    def __init__(self, argu_dict, event_argus_index_pair_dict, sentence):
        baseModel.__init__(self, argu_dict, event_argus_index_pair_dict, sentence)
        self.beginTime = argu_dict['Separation_BeginTime'] if ('Separation_BeginTime' in argu_dict) else ""
        self.beginTime_index_pair = event_argus_index_pair_dict['Separation_BeginTime'] if (
                'Separation_BeginTime' in event_argus_index_pair_dict) else None

        self.endTime = argu_dict['Separation_EndTime'] if ('Separation_EndTime' in argu_dict) else ""
        self.endTime_index_pair = event_argus_index_pair_dict['Separation_EndTime'] if (
                'Separation_EndTime' in event_argus_index_pair_dict) else None

        self.duration = argu_dict['Separation_Duration'] if ('Separation_Duration' in argu_dict) else ""
        self.duration_index_pair = event_argus_index_pair_dict['Separation_Duration'] if (
                'Separation_Duration' in event_argus_index_pair_dict) else None


    def __str__(self):
        str = self.negated+' '+((self.beginTime+' ') if len(self.beginTime)>0 else '')+self.trigger+' '+((self.endTime+' ') if len(self.endTime)>0 else '')+((self.duration+' ') if len(self.duration)>0 else '')
        return str.strip()

    def fit_arguments_by_spe(self, words, new_tags):
        self.beginTime = ''
        self.endTime = ''
        self.duration = ''
        for index, tag in enumerate(new_tags):
            if (tag.find('B_Separation_BeginTime') != -1):
                self.beginTime = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_Separation_BeginTime'):
                        self.beginTime = self.beginTime + words[theIndex]
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
        if (self.beginTime != None and self.beginTime != ''):
            score += 1
        if (self.endTime != None and self.endTime != ''):
            score += 1
        if (self.duration != None and self.duration != ''):
            score += 1
        return score

class DivorceLawsuit(baseModel):

    def __init__(self, argu_dict, event_argus_index_pair_dict, sentence):
        baseModel.__init__(self, argu_dict, event_argus_index_pair_dict, sentence)

        self.sueTime = argu_dict['DivorceLawsuit_SueTime'] if ('DivorceLawsuit_SueTime' in argu_dict) else ""
        self.sueTime_index_pair = event_argus_index_pair_dict['DivorceLawsuit_SueTime'] if (
                'DivorceLawsuit_SueTime' in event_argus_index_pair_dict) else None

        self.initiator = argu_dict['DivorceLawsuit_Initiator'] if ('DivorceLawsuit_Initiator' in argu_dict) else ""
        self.initiator_index_pair = event_argus_index_pair_dict['DivorceLawsuit_Initiator'] if (
                'DivorceLawsuit_Initiator' in event_argus_index_pair_dict) else None

        self.judgeTime = argu_dict['DivorceLawsuit_JudgeTime'] if ('DivorceLawsuit_JudgeTime' in argu_dict) else ""
        self.judgeTime_index_pair = event_argus_index_pair_dict['DivorceLawsuit_JudgeTime'] if (
                'DivorceLawsuit_JudgeTime' in event_argus_index_pair_dict) else None

        self.court = argu_dict['DivorceLawsuit_Court'] if ('DivorceLawsuit_Court' in argu_dict) else ""
        self.court_index_pair = event_argus_index_pair_dict['DivorceLawsuit_Court'] if (
                'DivorceLawsuit_Court' in event_argus_index_pair_dict) else None

        self.judgeDocument = argu_dict['DivorceLawsuit_JudgeDocument'] if ('DivorceLawsuit_JudgeDocument' in argu_dict) else ""
        self.judgeDocument_index_pair = event_argus_index_pair_dict['DivorceLawsuit_JudgeDocument'] if (
                'DivorceLawsuit_JudgeDocument' in event_argus_index_pair_dict) else None

        self.result = argu_dict['DivorceLawsuit_Result'] if ('DivorceLawsuit_Result' in argu_dict) else ""
        self.result_index_pair = event_argus_index_pair_dict['DivorceLawsuit_Result'] if (
                'DivorceLawsuit_Result' in event_argus_index_pair_dict) else None

    def fit_arguments_by_spe(self, words, new_tags):
        self.sueTime = ''
        self.initiator = ''
        self.court = ''
        self.judgeTime = ''
        self.judgeDocument = ''
        self.result = ''
        for index, tag in enumerate(new_tags):
            if (tag.find('B_DivorceLawsuit_SueTime') != -1):
                self.sueTime = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_DivorceLawsuit_SueTime'):
                        self.sueTime = self.sueTime + words[theIndex]
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
        if (self.sueTime != None and self.sueTime != ''):
            score += 1
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
class Wealth(baseModel):
    def __init__(self, argu_dict, event_argus_index_pair_dict, sentence):
        baseModel.__init__(self, argu_dict, event_argus_index_pair_dict, sentence)

        self.value = argu_dict['Wealth_Value'] if ('Wealth_Value' in argu_dict) else ""
        self.value_index_pair = event_argus_index_pair_dict['Wealth_Value'] if (
                'Wealth_Value' in event_argus_index_pair_dict) else None

        self.isCommon = argu_dict['Wealth_IsCommon'] if ('Wealth_IsCommon' in argu_dict) else ""
        self.isCommon_index_pair = event_argus_index_pair_dict['Wealth_IsCommon'] if (
                'Wealth_IsCommon' in event_argus_index_pair_dict) else None

        self.isPersonal = argu_dict['Wealth_IsPersonal'] if ('Wealth_IsPersonal' in argu_dict) else ""
        self.isPersonal_index_pair = event_argus_index_pair_dict['Wealth_IsPersonal'] if (
                'Wealth_IsPersonal' in event_argus_index_pair_dict) else None

        self.whose = argu_dict['Wealth_Whose'] if ('Wealth_Whose' in argu_dict) else ""
        self.whose_index_pair = event_argus_index_pair_dict['Wealth_Whose'] if (
                'Wealth_Whose' in event_argus_index_pair_dict) else None
    def __str__(self):
        return self.trigger

    def fit_arguments_by_spe(self, words, new_tags):
        self.value = ''
        self.isCommon = ''
        self.isPersonal = ''
        self.whose = ''
        for index, tag in enumerate(new_tags):
            if (tag.find('B_Wealth_Value') != -1):
                self.value = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_Wealth_Value'):
                        self.value = self.value + words[theIndex]
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
        if (self.value != None and self.value != ''):
            score += 1
        if (self.isCommon != None and self.isCommon != ''):
            score += 1
        if (self.isPersonal != None and self.isPersonal != ''):
            score += 1
        if (self.whose != None and self.whose != ''):
            score += 1
        return score
class Debt(baseModel):
    def __init__(self, argu_dict, event_argus_index_pair_dict, sentence):
        baseModel.__init__(self, argu_dict, event_argus_index_pair_dict, sentence)
        self.creditor = argu_dict['Debt_Creditor'] if ('Debt_Creditor' in argu_dict) else ""
        self.creditor_index_pair = event_argus_index_pair_dict['Debt_Creditor'] if (
                'Debt_Creditor' in event_argus_index_pair_dict) else None
        self.value = argu_dict['Debt_Value'] if ('Debt_Value' in argu_dict) else ""
        self.value_index_pair = event_argus_index_pair_dict['Debt_Value'] if (
                'Debt_Value' in event_argus_index_pair_dict) else None
    def __str__(self):
        return self.trigger

    def fit_arguments_by_spe(self, words, new_tags):
        self.value = ''
        self.creditor = ''
        for index, tag in enumerate(new_tags):
            if (tag.find('B_Debt_Value') != -1):
                self.value = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_Debt_Value'):
                        self.value = self.value + words[theIndex]
            if (tag.find('B_Debt_Creditor') != -1):
                self.creditor = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_Debt_Creditor'):
                        self.creditor = self.creditor + words[theIndex]

    def get_score(self):
        score = 1
        if (self.value != None and self.value != ''):
            score += 1
        if (self.creditor != None and self.creditor != ''):
            score += 1
        return score
class Credit(baseModel):
    def __init__(self, argu_dict, event_argus_index_pair_dict, sentence):
        baseModel.__init__(self, argu_dict, event_argus_index_pair_dict, sentence)
        self.debtor = argu_dict['Credit_Debtor'] if ('Credit_Debtor' in argu_dict) else ""
        self.debtor_index_pair = event_argus_index_pair_dict['Credit_Debtor'] if (
                'Credit_Debtor' in event_argus_index_pair_dict) else None

        self.value = argu_dict['Credit_Value'] if ('Credit_Value' in argu_dict) else ""
        self.value_index_pair = event_argus_index_pair_dict['Credit_Value'] if (
                'Credit_Value' in event_argus_index_pair_dict) else None
    def __str__(self):
        return self.trigger

    def fit_arguments_by_spe(self, words, new_tags):
        self.value = ''
        self.debtor = ''
        for index, tag in enumerate(new_tags):
            if (tag.find('B_Credit_Debtor') != -1):
                self.debtor = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_Credit_Debtor'):
                        self.debtor = self.debtor + words[theIndex]
            if (tag.find('B_Credit_Value') != -1):
                self.value = words[index]
                for theIndex in range(index + 1, len(new_tags)):
                    if (new_tags[theIndex] == 'I_Credit_Value'):
                        self.value = self.value + words[theIndex]

    def get_score(self):
        score = 1
        if (self.value != None and self.value != ''):
            score += 1
        if (self.debtor != None and self.debtor != ''):
            score += 1
        return score
if __name__ == '__main__':
    sys.exit(0)