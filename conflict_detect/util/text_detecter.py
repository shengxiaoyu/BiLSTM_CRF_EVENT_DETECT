#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'

import re
from conflict_detect.util.constant import NEGATION,REMARRY,FAMILY_CONFLICT_POSITIVE,GENDER,UP_TO_DATE
from model import myDate

#文本对齐、冲突检测工具类，判断规则和业务紧密相关
class text_detecter(object):

    def __init__(self):
        self.year_pattern = re.compile(r'\d{4}年')
        self.month_pattern = re.compile(r'\d{1,2}月')
        self.day_pattern = re.compile(r'\d{1,2}日')

    #时间冲突检测
    def check_time(self,time_a,time_b):
        if (self.direct_check_str(time_a,time_b)):
            return True
        year_a = self.year_pattern.findall(time_a)
        year_a = None if(year_a==None or len(year_a)==0) else year_a[0]
        month_a = self.month_pattern.findall(time_a)
        month_a = None if(month_a==None or len(month_a)==0) else month_a[0]
        day_a = self.day_pattern.findall(time_a)
        day_a = None if(day_a==None or len(day_a)==0) else day_a[0]

        year_b = self.year_pattern.findall(time_b)
        year_b = None if(year_b==None or len(year_b)==0) else year_b[0]
        month_b = self.month_pattern.findall(time_b)
        month_b = None if(month_b==None or len(month_b)==0) else month_b[0]
        day_b = self.day_pattern.findall(time_b)
        day_b = None if(day_b==None or len(day_b)==0) else day_b[0]

        if((year_a==None or year_b==None or year_a==year_b) and (month_a==None or month_b==None or month_b==month_a) and (day_a==None or day_b==None or day_b==day_a)):
            return True
        else:
            return False

    def format_time(self,time_a):
        if (time_a==None or time_a==''):
            return None
        for upToDate in UP_TO_DATE:
            if(time_a.find(upToDate)!=-1):
                return myDate.getNow()
        year_a = self.year_pattern.findall(time_a)
        year_a = None if(year_a==None or len(year_a)==0) else year_a[0][:-1]#去掉年字
        month_a = self.month_pattern.findall(time_a)
        month_a = None if(month_a==None or len(month_a)==0) else month_a[0][:-1]
        day_a = self.day_pattern.findall(time_a)
        day_a = None if(day_a==None or len(day_a)==0) else day_a[0][:-1]
        return myDate(year_a,month_a,day_a)

    #直接通过字符串是否包含或相等检测
    def direct_check_str(self,text_a,text_b):
        if (text_b == None or text_a == None):
            return True
        return text_b == text_a or text_a.find(text_b) != -1 or text_b.find(text_a) != -1


    #判断一个文本是正面还是负面,正面返回true，负面返回false
    def confirm_text_positive_negative(self, text):
        '''判断是正面肯定词还是负面否定词'''
        '''否定词返回false，肯定词返回true'''
        for val in NEGATION['NEGATIVE']:
            if(text.find(val)!=-1):
                return False
        for val in NEGATION['POSITIVE']:
            if(text.find(val)!=-1):
                return True
        print('无法匹配否定词肯定词')
        return True

    #如果是初婚返回true，如果是二婚返回false
    def confirm_first_marry_or_remarry(self,text):
        for val in REMARRY['REMARRY']:
            if(val==text):
                return False
        for val in REMARRY['FIRST_MARRY']:
            if(val==text):
                return True
        print('无法匹配初婚再婚')

    #判断一个家庭矛盾的描述语是正面还是负面的，当前方法是收集正面表达语，判断text中是否包含正面语，包含则说明
    #为正面，反之认为是负面
    def confirm_family_conflict_positive_or_negative(self,text):
        for positive_text in FAMILY_CONFLICT_POSITIVE:
            if(text.find(positive_text)!=-1):
                return True
        return False

    #检测传入性别，如果相同则返回true，否则返回false
    def check_gender(self,gender_a,gender_b):
        if (self.direct_check_str(gender_a, gender_b)):
            return True
        _gender_a = True
        _gender_b = True
        for male in GENDER['M']:
            if(gender_a.find(male)!=-1):
                _gender_a = True
            if(gender_b.find(male)!=-1):
                _gender_b = True
        for fmale in GENDER['F']:
            if(gender_a.find(fmale)!=-1):
                _gender_a = False
            if(gender_b.find(fmale)!=-1):
                _gender_b = False

        return _gender_a and _gender_b

    def normalize_name(self,event,attr):
        '''格式化名称,结果只为原告或被告'''
        if(not hasattr(event,attr)):
            return None
        attr_entity = getattr(event,attr)
        val = attr_entity.value
        if(val.find('原告')):
            return '原告'
        if(val.find('被告')):
            return '被告'
        return self.normalize_me(event,attr)

    def normalize_me(self,event,attr):
        '''格式化’我‘字，转换为事件的speaker'''
        if (not hasattr(event, attr)):
            return None
        attr_entity = getattr(event,attr)
        val = attr_entity.value
        if (val.find('我')!=-1 or val.find('自己')!=-1):
            return event.speaker.value
        return val


if __name__ == '__main__':
    pass