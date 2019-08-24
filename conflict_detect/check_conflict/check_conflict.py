#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'
from conflict_detect.event_alignment.alignmenter import alignmenter
from conflict_detect.util.text_detecter import text_detecter
from model import myDate

#事件冲突检测器，检测两个对齐事件是否冲突
class checker_conflict(object):
    def __init__(self):
        self.event_alignmenter = alignmenter()
        self.detecter = text_detecter()

    '''检测两个事件是否冲突，如果冲突返回True，反之返回False'''
    def check(self,event_a,event_b):
        '''判断事件是否冲突，两种主要策略：'''
        ''' 1、语义否定；2、否定词否定，'''
        '''否定词可以有两个，事件a,b各一个，语义否定需要根据事件分别特殊对待'''
        '''判断的时候，我们首先对对齐事件的属性进行语义否定的检测，只有当事件未能检测出矛盾属性，即蕴含关系时，再去通过否定词检测'''


        '''首先判断事件是否是同一类事件'''
        if(event_a.type!=event_b.type):
            return False

        '''然后判断事件是否是指同一件事件'''
        isAlignment = self.event_alignmenter.alignment(event_a,event_b)
        if(not isAlignment):
            return False

        '''判断事件描述是否有冲突'''
        type_name = event_a.type.value
        type_method = getattr(self,'_check_'+type_name)
        return type_method(event_a,event_b)


    '''单个事件：有否定词返回false，没有否定词或者有肯定词，则返回true'''
    def _check_negated(self,event):
        res = True
        if(hasattr(event,'Negation')):
            val = event.Negation
            '''判断是肯定语义还是否定语义'''
            if(not self.detecter.confirm_text_positive_negative(val.value)):
                res = False
        return res

    '''判断两个相识事件是否冲突，冲突返回True，反之返回False'''
    def _check_Know(self,event_a,event_b):
        '''判断相识事件，相识事件是唯一性事件，因此如果时间不同则发生矛盾'''

        #先判断时间是否相同
        res = self.detecter.check_time(event_a.Time.value if hasattr(event_a,'Time') else None,event_b.Time.value if hasattr(event_b,'Time') else None)
        '''考虑是否有否定词/肯定词,这里对两个事件都出现否定词不做考虑，因为一般只有一个人对另一个人否定或肯定，不存在两个都否定的情况，而且及时两个都否定同一个事件，按照逻辑也应该负负得正'''
        #时间相同则考虑否定词
        if(res):
            res = (res and self._check_negated(event_a))
            res = (res and self._check_negated(event_b))

        #最后如果结果res为true，说明时间相互蕴含，而且不含有否定词，因此事件不冲突，因此返回False
        #如果结果为false，说明有时间不同或者有否定词，因此有冲突返回true
        return not res

    '''判断两个相恋事件是否冲突，冲突返回True，反之返回False'''
    def _check_BeInLove(self,event_a,event_b):
        '''相恋事件同相识事件'''
        res = self.detecter.check_time(event_a.Time.value if hasattr(event_a,'Time') else None,event_b.Time.value if hasattr(event_b,'Time') else None)
        if (res):
            res = (res and self._check_negated(event_a))
            res = (res and self._check_negated(event_b))
        return not res

    '''判断两个结婚事件是否冲突，冲突返回True，反之返回False'''
    def _check_Marry(self,event_a,event_b):
        '''根据是否有否定词判断，因为两个结婚事件如果对齐则说明结婚时间一定相同'''
        res = self._check_negated(event_a) and self._check_negated(event_b)
        return not res

    '''判断两个再婚事件是否冲突，冲突返回True，反之返回False'''
    def _check_Remarry(self,event_a,event_b):
        #两个描述相同则说明不冲突
        res = (self.detecter.confirm_first_marry_or_remarry(event_a.trigger.value) == self.detecter.confirm_first_marry_or_remarry(event_b.trigger.value))
        if(res):
            res = (res and self._check_negated(event_a))
            res = (res and self._check_negated(event_b))
        return  not res

    '''判断两个家庭矛盾事件是否冲突，冲突返回True，反之返回False'''
    def _check_FamilyConflict(self,event_a,event_b):
        res = (self.detecter.confirm_family_conflict_positive_or_negative(event_a.trigger.value) == self.detecter.confirm_family_conflict_positive_or_negative(event_b.trigger.value))
        if(res):
            res = res and self._check_negated(event_a)
            res = res and self._check_negated(event_b)
        return not res

    '''判断两个生育事件是否冲突，冲突返回True，反之返回False'''
    def _check_Bear(self,event_a,event_b):
        '''检测生育事件，判断时间、性别是否能相互蕴含，如果蕴含再判断是否含有否定词'''
        res = self.detecter.check_time(event_a.Time.value if hasattr(event_a,'Time') else None,event_b.Time.value if hasattr(event_b,'Time') else None)
        if(res):
            res = self.detecter.check_gender(event_a.Gender.value if hasattr(event_a,'Gender') else None,event_b.Gender.value if hasattr(event_b,'Gender') else None)
            if(res):
                res = self._check_negated(event_a)
                res = (res and self._check_negated(event_b))
        return  not res

    '''判断两个财产事件是否冲突，冲突返回True，反之返回False'''
    def _check_Wealth(self,event_a,event_b):
        '''财产的判断主要是判断两个财产描述中，财产的所属人是否冲突：个人财产或共同财产'''

        #a认为是共同财产，不是个人财产
        if((hasattr(event_a,'IsCommon') and self._check_negated(event_a)) or (hasattr(event_a,'IsPersonal') and not self._check_negated(event_a))):

            #b认为不是共同财产，冲突
            if(hasattr(event_b,'IsCommon')):
                if(not self._check_negated(event_b)):
                    return True

            #b认为是个人财产,冲突
            if(hasattr(event_b,'IsPersonal') and self._check_negated(event_b)):
                return True

        #a认为是个人财产，不是共同财产
        if((hasattr(event_a,'IsCommon') and not self._check_negated(event_a)) or (hasattr(event_a,'IsPersonal') and self._check_negated(event_a))):
            #b认为是共同财产，冲突
            if(hasattr(event_b,'IsCommon') and self._check_negated(event_b)):
                return True

            #b认为是个人财产，但所属人不同
            if(hasattr(event_b,'IsPersonal')):
                if(self.detecter.normalize_name(event_a,'Whose')!=self.detecter.normalize_name(event_b,'Whose')):
                    '''所属人不同'''
                    return True
                else:
                    if(not self._check_negated(event_b)):
                        '''所属人相同，但是有否定词'''
                        return True
        #判断财产的价值描述是否相同
        if(hasattr(event_a,'Value') and hasattr(event_b,'Value') and (not self.detecter.direct_check_str(event_a.Value.value,event_b.Value.value))):
            return True

        #如果以上都没有，则直接判断是否某个事件含有否定词
        if(not self._check_negated(event_a) or not self._check_negated(event_b)):
            return  True
        return False

    '''判断两个家暴事件是否冲突，冲突返回True，反之返回False'''
    def _check_DomesticViolence(self,event_a,event_b):
        res = self._check_negated(event_a)
        res = res and self._check_negated(event_b)
        return not res

    '''判断两个坏习惯事件是否冲突，冲突返回True，反之返回False'''
    def _check_BadHabit(self,event_a,event_b):
        res = self._check_negated(event_a)
        res = res and self._check_negated(event_b)
        return not res

    '''判断两个出轨事件是否冲突，冲突返回True，反之返回False'''
    def _check_Derailed(self,event_a,event_b):
        res = self._check_negated(event_a)
        res = res and self._check_negated(event_b)
        return not res

    '''判断两个分居事件是否冲突，冲突则返回True，反之返回False'''
    def _check_Separation(self,event_a,event_b):
        #判断截至时间是否能相互蕴含
        res = True

        #先通过起止时间段判断，如果有交集且不相等则冲突
        begin_time_a = None
        end_time_a = None
        begin_time_b = None
        end_time_b = None

        if(hasattr(event_a,'BeginTime')):
            begin_time_a = self.detecter.format_time(event_a.BeginTime.value)
        if(hasattr(event_a,'EndTime')):
            end_time_a = self.detecter.format_time(event_a.EndTime.value)
        if(hasattr(event_b,'BeginTime')):
            begin_time_b = self.detecter.format_time(event_b.BeginTime.value)
        if(hasattr(event_b,'EndTime')):
            end_time_b = self.detecter.format_time(event_b.EndTime.value)

        #两个时间段有明显的分割，则说明不冲突
        if(myDate.compareTo(end_time_a,begin_time_b)==-1 or myDate.compareTo(end_time_b,begin_time_a)==-1):
            return False

        #开始时间和截止时间不一样，冲突
        if(myDate.compareTo(begin_time_a,begin_time_b)!=0 or myDate.compareTo(end_time_b,end_time_a)!=0):
            return True

        #判断持续时间属性是否冲突
        if(not self.detecter.direct_check_str(event_a.Duration.value if hasattr(event_a,'Duration') else None,event_b.Duration.value if hasattr(event_b,'Duration') else None)):
            return False

        #走到这里说明通过时间节点没能判断出来，则判断是否有否定词
        if((not self._check_negated(event_a)) or (not self._check_negated(event_b))):
            return True
        return False

    '''判断两个离婚诉讼事件是否冲突，冲突则返回True，反之返回False'''
    def _check_DivorceLawsuit(self,event_a,event_b):
        #只需判断是否含有否定词
        res = self._check_negated(event_a) and self._check_negated(event_b)
        return not res

    '''判断两个债务事件是否冲突，冲突返回True，反之返回False'''
    def _check_Debt(self,event_a,event_b):
        '''判断价值和否定词'''
        res = True
        if(self.detecter.direct_check_str(event_a.Value.value if hasattr(event_a,'Value') else None,event_b.Value.value if hasattr(event_b,'Value') else None)):
            res = self._check_negated(event_a) and self._check_negated(event_b)
        return not res
    '''判断两个债权事件是否冲突，冲突返回True，反之返回False'''
    def _check_Credit(self,event_a,event_b):
        '''判断价值和否定词'''
        res = True
        if (self.detecter.direct_check_str(event_a.Value.value if hasattr(event_a, 'Value') else None,
                                           event_b.Value.value if hasattr(event_b, 'Value') else None)):
            res = self._check_negated(event_a) and self._check_negated(event_b)
        return not res
if __name__ == '__main__':
    pass