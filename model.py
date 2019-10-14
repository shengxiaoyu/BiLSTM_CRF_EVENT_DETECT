#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'
from enum import Enum
import datetime
from event_dectect.Second_For_Fine_Tags import config_center as NEW_CONFIG
class Entity(object):
    def __init__(self, id, type, begin, end, value):
        self.id = id.strip()
        self.type = type.strip()
        self.begin = int(begin)
        self.end = int(end)
        self.value = value.strip()
    def _simple_json(self):
        json_str = {}
        json_str['begin'] = self.begin
        json_str['end'] = self.end
        json_str['value'] = self.value
        return json_str

class Event(object):

    @staticmethod
    def _formEvent(id, tags, words, index_pairs, sentence, speaker):
        event_type = None
        trigger = None
        argus = {}
        for tag, word, index_pair in zip(tags, words, index_pairs):
            if (tag in NEW_CONFIG.NEW_TRIGGER_TAGs):
                '''触发词标签，确定触发词和事件类型'''
                if(event_type==None):
                    type_str = tag[2:-8]
                    event_type = event_types.get_event_type(type_str)
                    trigger = Entity("",type_str,index_pair[0],index_pair[1],word)
                else:
                    '''I_开头'''
                    trigger.end = index_pair[1]
                    trigger.value = sentence[trigger.begin:trigger.end]
            if (tag in NEW_CONFIG.NEW_ARGU_TAGs and tag != '<pad>' and tag != 'O'):
                argu_type_str = tag[tag.rfind('_')+1:]
                if(argu_type_str not in argus.keys()):
                    argu = Entity("",argu_type_str,index_pair[0],index_pair[1],word)
                    argus[argu_type_str] = argu
                else:
                    argu = argus[argu_type_str]
                    argu.end = index_pair[1]
                    argu.value = sentence[argu.begin:argu.end]
                    argus[argu_type_str] = argu
        if(event_type!=None and trigger!=None):
            event = Event(id,event_type,trigger)
            event.speaker = speaker
            event._addSent(sentence)
            for argu_type in argus.keys():
                event._setArgu(argu_type, argus[argu_type])
            return event
        return None



    def __init__(self, id, type, trigger):
        self.id = id.strip()
        self.type = type
        self.trigger = trigger
        self.sentence = None
        self.begin_index = 0
        self.speaker = None
    def _setArgu(self, argu_name, argu_value):
        setattr(self,argu_name,argu_value)
    def _addSent(self, sent):
        if(self.sentence ==None):
            self.sentence = sent
        else:
            self.sentence += sent

    def __str__(self):
        res = 'id:'+self.id+', '
        if(self.type==None):
            print('Pause')
        res += 'type:'+self.type.value+', '
        if(hasattr(self,'Negation')):
            res+='Negation:'+self.Negation.value
        res += 'trigger:'+self.trigger.value+', '
        if(hasattr(self,'Time')):
            res += 'Time:'+self.Time.value+', '
        if(hasattr(self,'Participant')):
            res += 'Participant:'+self.Participant.value+', '
        if(hasattr(self,'ChildName')):
            res += 'ChildName:'+self.ChildName.value+', '
        if(hasattr(self,'DateOfBirth')):
            res += 'DateOfBirth:'+self.DateOfBirth.value+', '
        if(hasattr(self,'Gender')):
            res += 'Gender:'+self.Gender.value+', '
        if(hasattr(self,'Age')):
            res += 'Age:'+self.Age.value+', '
        if(hasattr(self,'Perpetrators')):
            res += 'Perpetrators:'+self.Perpetrators.value
        if(hasattr(self,'Victim')):
            res += 'Victim:'+self.Victim.value
        if(hasattr(self,'Derailer')):
            res += 'Derailer:'+self.Derailer.value+', '
        if(hasattr(self,'BeginTime')):
            res += 'BeginTime:'+self.BeginTime.value+', '
        if(hasattr(self,'EndTime')):
            res += 'EndTime:'+self.EndTime.value+', '
        if(hasattr(self,'Duration')):
            res += 'Duration:'+self.Duration.value+', '
        if(hasattr(self,'SueTime')):
            res += 'SueTime:'+self.SueTime.value+', '
        if(hasattr(self,'Initiator')):
            res += 'Initiator:'+self.Initiator.value+', '
        if(hasattr(self,'Court')):
            res += 'Court:'+self.Court.value+', '
        if(hasattr(self,'Result')):
            res += 'Result:'+self.Result.value+', '
        if(hasattr(self,'JudgeTime')):
            res += 'JudgeTime:'+self.JudgeTime.value+', '
        if(hasattr(self,'JudgeDocument')):
            res += 'JudgeDocument'+self.JudgeDocument.value+', '
        if(hasattr(self,'Value')):
            res += 'Value:'+self.Value.value+', '
        if(hasattr(self,'IsPersonal')):
            res += 'IsPersonal:'+self.IsPersonal.value+', '
        if(hasattr(self,'Whose')):
            res += 'Whose:'+self.Whose.value+', '
        if(hasattr(self,'IsCommon')):
            res += 'IsCommon:'+self.IsCommon.value+', '
        if(hasattr(self,'Creditor')):
            res += 'Creditor:'+self.Creditor.value+', '
        if(hasattr(self,'Debtor')):
            res += 'Debtor:'+self.Debtor.value+', '
        res =res.strip()
        if(res.endswith(',')):
            res = res[:-1]
        return res

    def _obj_to_json(self):
        json_str = {}
        for att_str in self.__dir__():
            if(not att_str.startswith('_')):
                if(att_str=='id' or att_str=='sentence'):
                    json_str[att_str] = getattr(self,att_str)
                elif(att_str=='begin_index'):
                    json_str[att_str] = str(getattr(self,att_str))
                elif(att_str=='type'):
                    json_str[att_str] = getattr(self,att_str).value
                elif(att_str=='speaker'):
                    json_str[att_str] = getattr(self,att_str).value
                else:
                    json_str[att_str] = getattr(self,att_str)._simple_json()

         # json_str.keys()
        return json_str
class Relation(object):
    def __init__(self,id, type, arg1, arg2):
        self.id = id
        self.type = type

        #事件id，保证小id为arg1,大id为arg2
        id_index_1 = int(arg1.id[1:])
        id_index_2 = int(arg2.id[1:])

        if(id_index_1>id_index_2):
            self.arg1 = arg2
            self.arg2 = arg1
        else:
            self.arg1 = arg1
            self.arg2 = arg2

    def __eq__(self, other):
        if(other==None):
            return False
        if(self.arg1==other.arg1 and self.arg2==other.arg2):
            return True
        return False

    def _obj_to_json(self):
        json_str = {}
        json_str['type'] = self.type
        json_str['arg1'] = self.arg1.id
        json_str['arg2'] = self.arg2.id
        return json_str
class event_types(Enum):
    KNOW = 'Know'
    BE_IN_LOVE = 'BeInLove'
    MARRY = 'Marry'
    REMARRY = 'Remarry'
    BE_BORN = 'Bear'
    FAMILY_CONFLICT = 'FamilyConflict'
    DOMESTIC_VIOLENCE = 'DomesticViolence'
    BAD_HABIT = 'BadHabit'
    DERAILED = 'Derailed'
    SEPARATION = 'Separation'
    DIVORCE_LAWSUIT = 'DivorceLawsuit'
    WEALTH = 'Wealth'
    DEBT = 'Debt'
    CREDIT = 'Credit'

    def get_event_type(text):
        text = text.strip()
        for item in event_types:
            if(item.value==text):
                return item


class parties(Enum):
    '''原告'''
    PLAINTIFF = '原告'
    '''被告'''
    DEFENDANT = '被告'


class habit_types(Enum):
    GAMBLING = '赌博'
    WINE = '酗酒'
    DRUG = '吸毒'
    '''嫖娼'''
    SEXUAL = '嫖娼'
    '''传销'''
    MLM = '传销'
    '''网瘾'''
    NETWORK = '网瘾'
    '''偷抢'''
    THEFT = '偷抢'
    '''打架斗殴'''
    FIGHT = '斗殴'
    '''诈骗'''
    FRAUD = '诈骗'


class myDate(object):

    @staticmethod
    def getNow():
        now = datetime.datetime.now()
        return myDate(now.year,now.month,now.day)

    def __init__(self,year,month,day):
        self.year = None if year==None else int(year)
        self.month = None if month==None else int(month)
        self.day = None if day==None else int(day)

    '''时间的比较，如果可以相互蕴含则返回0，如果大返回1，小返回-1'''
    @staticmethod
    def compareTo(date_a, date_b):
        if(date_a==None or date_b==None):
            return 0
        if(date_a.year!=None and date_b.year!=None):
            if(date_a.year>date_b.year):
                return 1
            elif(date_a.year<date_b.year):
                return -1
        if(date_a.month!=None and date_b.month!=None):
            if(date_a.month>date_b.month):
                return 1
            elif(date_a.month<date_b.month):
                return -1
        if(date_a.day!=None and date_b.day!=None):
            if(date_a.day>date_b.day):
                return 1
            elif(date_a.day<date_b.day):
                return -1
        return 0

if __name__ == '__main__':
    var = event_types.get_event_type('Marry')
    print(var)