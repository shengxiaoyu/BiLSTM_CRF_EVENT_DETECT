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


#相识事实有一个时间参数，往前找
class Know(baseModel):
    def __init__(self,argu_dict,event_argus_index_pair_dict,sentence):
        baseModel.__init__(self,argu_dict,event_argus_index_pair_dict,sentence)
        self.time =argu_dict['Know_Time'] if('Know_Time' in argu_dict) else ""
        self.time_index_pair =event_argus_index_pair_dict['Know_Time'] if('Know_Time' in event_argus_index_pair_dict) else None

    def __str__(self):
        str = self.negated+' '+self.time+' '+self.trigger
        return str.strip()

class BeInLove(baseModel):
    def __init__(self, argu_dict, event_argus_index_pair_dict, sentence):
        baseModel.__init__(self, argu_dict, event_argus_index_pair_dict, sentence)
        self.time = argu_dict['BeInLove_Time'] if ('BeInLove_Time' in argu_dict) else ""
        self.time_index_pair = event_argus_index_pair_dict['BeInLove_Time'] if (
                    'BeInLove_Time' in event_argus_index_pair_dict) else None

    def __str__(self):
        str =  self.negated+' '+self.time + ' ' + self.trigger
        return str.strip()

class Marray(baseModel):
    def __init__(self, argu_dict, event_argus_index_pair_dict, sentence):
        baseModel.__init__(self, argu_dict, event_argus_index_pair_dict, sentence)
        self.time = argu_dict['Marry_Time'] if ('Marry_Time' in argu_dict) else ""
        self.time_index_pair = event_argus_index_pair_dict['Marry_Time'] if (
                'Marry_Time' in event_argus_index_pair_dict) else None

    def __str__(self):
        str =  self.negated+' '+self.time + ' ' + self.trigger
        return str.strip()

class Remarray(baseModel):
    def __init__(self, argu_dict, event_argus_index_pair_dict, sentence):
        baseModel.__init__(self, argu_dict, event_argus_index_pair_dict, sentence)
        self.participant = argu_dict['Remarry_Participant'] if ('Remarry_Participant' in argu_dict) else ""
        self.participant_index_pair = event_argus_index_pair_dict['Remarry_Participant'] if ('Remarry_Participant' in event_argus_index_pair_dict) else None

    def __str__(self):
        str =  self.negated+' '+self.participant+' '+self.trigger
        return str.strip()


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

class FamilyConflict(baseModel):
    def __init__(self, argu_dict, event_argus_index_pair_dict, sentence):
        baseModel.__init__(self, argu_dict, event_argus_index_pair_dict, sentence)
    def __str__(self):
        return self.trigger

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

class BadHabit(baseModel):
    def __init__(self, argu_dict, event_argus_index_pair_dict, sentence):
        baseModel.__init__(self, argu_dict, event_argus_index_pair_dict, sentence)

        self.person = argu_dict['BadHabit_Participant'] if ('BadHabit_Participant' in argu_dict) else ""
        self.person_index_pair = event_argus_index_pair_dict['BadHabit_Participant'] if (
                'BadHabit_Participant' in event_argus_index_pair_dict) else None

    def __str__(self):
        str = self.negated+' '+((self.person+' ') if len(self.person)>0 else '')+self.trigger
        return str.strip()

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

if __name__ == '__main__':
    sys.exit(0)