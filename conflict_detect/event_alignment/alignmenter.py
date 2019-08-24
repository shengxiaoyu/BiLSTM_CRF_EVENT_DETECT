#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'

from conflict_detect.edit_distance_similar.similar_calculator import EditDistanceSimilarCalculator
import re
from model import habit_types,event_types
from conflict_detect.util.text_detecter import text_detecter


#事件对齐类，用于判断两个事件是否描述同一事件
class alignmenter(object):

    def __init__(self):
        self.similar_calculator = EditDistanceSimilarCalculator()
        self.family_conflict_similar_threshold = 0.45
        self.wealth_similar_threshold = 0.5
        self.checker = text_detecter()

        self.habits = {}
        self.habits[habit_types.GAMBLING] = ['打牌','打麻将','赌博','搓麻将','打赌','六合彩','赌光','赌马','赌钱','赌球','好赌','买马','嗜赌','百家乐','玩麻将','吃喝嫖赌']
        self.habits[habit_types.WINE] = ['喝酒','酗酒','醉酒','酒驾','嗜酒','嗜烟酒','酒']
        self.habits[habit_types.DRUG] = ['吸毒','吸食毒品','涉毒上瘾']
        self.habits[habit_types.SEXUAL] = ['嫖娼','找小姐']
        self.habits[habit_types.MLM] = ['传销']
        self.habits[habit_types.NETWORK] = ['网络','上网','网吧','网瘾']
        self.habits[habit_types.THEFT] = ['盗窃','偷','抢劫']
        self.habits[habit_types.FIGHT] = ['打架','斗殴']
        self.habits[habit_types.FRAUD] = ['诈骗','骗取钱财']

    def alignment(self,event_a,event_b):
        '''所有的比较满足：如果其中一个为空，则返回False'''
        if(event_a.type!=event_b.type):
            return False

        '''相识、相恋事件是唯一性事件'''
        if(event_a.type==event_types.KNOW or event_a.type==event_types.BE_IN_LOVE):
            return True

        if(event_a.type==event_types.REMARRY):
            '''再婚比较主体'''
            return self._alignment_by_wealther_equal(self.checker.normalize_name(event_a,'Participant'),self.checker.normalize_name(event_b,'Participant'))

        if(event_a.type==event_types.FAMILY_CONFLICT):
            '''家庭矛盾比较触发词的相似度'''
            return self._alignment_family_conflict(event_a.trigger.value,event_b.trigger.value)

        if(event_a.type==event_types.WEALTH):
            '''财产比较触发词的相似度'''
            res = self._alignment_wealth(event_a.trigger.value,event_b.trigger.value)
            return res
        if(event_a.type==event_types.BE_BORN):
            '''生育先比较孩子的姓名。如果孩子姓名不存在则比较性别'''
            if(hasattr(event_a,'ChildName') and hasattr(event_b,'ChildName')):
                return self._alignment_by_wealther_equal(event_a.ChildName.value,event_b.ChildName.value)
            else:
                return self._alignment_by_wealther_equal(event_a.Gender.value if hasattr(event_a,'Gender') else None,event_b.Gender.value if hasattr(event_b,'Gender') else None)
        if(event_a.type==event_types.MARRY):
            '''复婚比较时间'''
            return self.alignment_time(event_a.Time.value if hasattr(event_a, 'Time') else None, event_b.Time.value if hasattr(event_b, 'Time') else None)

        if(event_a.type==event_types.DOMESTIC_VIOLENCE):
            '''家暴比较家暴人、家暴对象、家暴时间'''
            return self._alignment_by_wealther_equal(self.checker.normalize_name(event_a,'Perpetrators'),self.checker.normalize_name(event_b,'Perpetrators')) \
                   and self._alignment_by_wealther_equal(self.checker.normalize_name(event_a,'Victim'),self.checker.normalize_name(event_b,'Victim')) \
                   and self.alignment_time(event_a.Time.value if hasattr(event_a, 'Time') else None, event_b.Time.value if hasattr(event_b, 'Time') else None)
        if(event_a.type==event_types.BAD_HABIT):
            '''坏习惯比较主体和习惯类别'''
            return self._alignment_by_wealther_equal(self.checker.normalize_name(event_a,'Participant'),self.checker.normalize_name(event_b,'Participant')) and \
                   self._alignment_habit(event_a.trigger.value,event_b.trigger.value)
        if(event_a.type==event_types.DERAILED):
            '''出轨比较出轨人和时间'''
            return self._alignment_by_wealther_equal(self.checker.normalize_name(event_a,'Derailer'),self.checker.normalize_name(event_b,'Derailer')) and \
                   self.alignment_time(event_a.Time.value if hasattr(event_a, 'Time') else None, event_b.Time.value if hasattr(event_b, 'Time') else None)
        if(event_a.type==event_types.SEPARATION):
            '''把分居看作是唯一事件，都能对齐，在冲突检测时，如果两个分居事件的持续时间段有交集，则说明冲突'''
            # if(hasattr(event_a))
            # return self.alignment_time(event_a.BeginTime.value if hasattr(event_a, 'BeginTime') else None, event_b.BeginTime.value if hasattr(event_b, 'BeginTime') else None)
            return True
        if(event_a.type==event_types.DIVORCE_LAWSUIT):
            '''离婚诉讼比较起诉时间和起诉人'''
            return self.alignment_time(event_a.SueTime.value if hasattr(event_a, 'SueTime') else None, event_b.SueTime.value if hasattr(event_b, 'SueTime') else None) and \
                   self._alignment_by_wealther_equal(self.checker.normalize_name(event_a,'Initiator'),self.checker.normalize_name(event_b,'Initiator'))
        if(event_a.type==event_types.DEBT):
            '''比较债务人'''
            return self._alignment_by_wealther_equal(event_a.Creditor.value if hasattr(event_a,'Creditor') else None,event_b.Creditor.value if hasattr(event_b,'Creditor') else None)
        if(event_a.type==event_types.CREDIT):
            '''比较债权人'''
            return self._alignment_by_wealther_equal(event_a.Debtor.value if hasattr(event_a,'Debtor') else None,event_b.Debtor.value if hasattr(event_b,'Debtor') else None)


    def _alignment_family_conflict(self,text_a,text_b):
        '''event_a,event_b must be family_conflict'''
        if(self._wealther_contain_or_equal_or_none(text_a,text_b)):
            return True
        similarity = self.similar_calculator.cal(text_a,text_b)
        if(similarity>=self.family_conflict_similar_threshold):
            return True
        else:
            return False

    def _alignment_by_wealther_equal(self,text_a,text_b):
        '''directly judge wealther equal'''
        return self._wealther_contain_or_equal_or_none(text_b,text_a)

    def _alignment_wealth(self,text_a,text_b):
        if (self._wealther_contain_or_equal_or_none(text_a, text_b)):
            return True
        similarity = self.similar_calculator.cal(text_a,text_b)
        if(similarity>=self.wealth_similar_threshold):
            return True
        else:
            return False

    def _alignment_habit(self,text_a,text_b):
        if (self._wealther_contain_or_equal_or_none(text_a, text_b)):
            return True
        a_type = None
        b_type = None
        for (key,values) in self.habits.items():
            for value in values:
                if(text_a.find(value)!=-1):
                    a_type = key
                if(text_b.find(value)!=-1):
                    b_type = key
            if(a_type!=None or b_type!=None):
                break
        return a_type==b_type

    def _alignment_remarry(self,text_a,text_b):
        if (self._wealther_contain_or_equal_or_none(text_a,text_b)):
            return True

    def alignment_time(self, time_a, time_b):
        return self.checker.check_time(time_a, time_b)
    def _wealther_contain_or_equal_or_none(self,text_a,text_b):
        return self.checker.direct_check_str(text_a,text_b)

if __name__ == '__main__':
    str = '2014年月1日'
    month_pattern = re.compile('([0-9]{1,2})月')
    month = month_pattern.findall(str)
    month = None if(month==None or len(month)==0) else month[0]
    print(month)
    pass