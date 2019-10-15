#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

__doc__ = 'description'
__author__ = '13314409603@163.com'
from pyltp import SentenceSplitter
from model import parties
from event_dectect.Second_For_Fine_Tags import word2vec_lstm_crf_argu_match as second
from event_dectect.Config.config_parser import getParser
from event_dectect.First_For_Commo_Tags import config_center as CONFIG
from event_dectect.Second_For_Fine_Tags import config_center as NEW_CONFIG
from event_dectect.First_For_Commo_Tags import word2vec_lstm_crf_ed as first
from model import Event


class Extractor(object):
    def __init__(self,first_output_path=None,second_output_path=None):
        print('构造参数')
        self.FLAGS = getParser()
        self.FLAGS.ifTrain = False
        self.FLAGS.ifTest = False
        self.FLAGS.ifPredict = True
        self.first_output_path = first_output_path
        self.second_output_path = second_output_path
        CONFIG.init(self.FLAGS.root_dir)
        NEW_CONFIG.initNewTags(os.path.join(self.FLAGS.root_dir, 'full_trigger_labels.txt'),
                               os.path.join(self.FLAGS.root_dir, 'full_argu_labels.txt'))
        self.sentenceSplitter = SentenceSplitter()

    '''传入文本段抽取'''
    def extractor(self,paragraph):
        # 调用预测接口

        sentences = []
        speakers = []
        isYg = True
        for sentence in self.sentenceSplitter.split(paragraph):
            if(sentence.find('诉称')!=-1):
                isYg = True
            if(sentence.find('辩称')!=-1):
                isYg = False
            hasTrigger = ifContainTrigger(sentence)
            if (not hasTrigger):
                print('该句子中无关注事实：' + sentence)
            else:
                sentences.append(sentence)
                if(isYg):
                    speakers.append(parties.PLAINTIFF)
                else:
                    speakers.append(parties.DEFENDANT)
        print('第一个模型预测')
        if(len(sentences)==0):
            print("整个抽取文本无关注事实")
            return []
        words_list,first_tags_list,index_pairs_list = first.main(self.FLAGS, sentences,output_path=self.first_output_path)
        print('第二个模型预测')
        if(self.FLAGS.second_label):
            words_list,second_tags_list,index_pairs_list,sentences,speakers = self.second_label([words_list,first_tags_list,index_pairs_list,sentences,speakers])
        else:
            words_list,second_tags_list,index_pairs_list,sentences,speakers = self.argu_match_base(words_list,first_tags_list,index_pairs_list,sentences,speakers)


        events = []
        id_index = 0
        for tags, words,index_pairs,sentence,speaker in zip(second_tags_list, words_list,index_pairs_list,sentences,speakers):
            event = Event._formEvent('E' + str(id_index), tags, words, index_pairs, sentence, speaker)
            if(event):
                events.append(event)
                id_index += 1
        print('事件抽取完成：\n'+'\n'.join([str(event) for event in events]))
        return events

    '''使用两阶段标记的方式'''
    def second_label(self,words_firstTags_indxPairs_sentences_speaker):
        return second.main(self.FLAGS,words_firstTags_indxPairs_sentences_speaker=words_firstTags_indxPairs_sentences_speaker,output_path=self.second_output_path)

    '''第一层模型预测之后基于规则匹配'''
    def argu_match_base(self,words_list,first_tags_list,index_pairs_list,sentences,speakers):
        words_lists=[]
        second_tags_lists=[]
        index_pairs_lists=[]
        sentencess=[]
        speakerss = []
        
        for words, tags, words_in_sentence_index_pair, sentence, speaker in zip(words_list, first_tags_list,
                                                                       index_pairs_list, sentences,speakers):
            words_list, second_tags_list, index_pairs_list, sentences, speakers = self.__get_final_tags_from_base_tags(words, tags, words_in_sentence_index_pair, sentence,speaker)
            words_lists.extend(words_list)
            second_tags_lists.extend(second_tags_list)
            index_pairs_lists.extend(index_pairs_list)
            sentencess.extend(sentences)
            speakerss.extend(speakers)
        return words_lists,second_tags_lists,index_pairs_lists,sentencess,speakerss

    def __get_final_tags_from_base_tags(self, words,tags,words_in_sentence_index_pair, sentence,speaker):
        # 获取触发词tag
        triggers = CONFIG.TRIGGER_TAGs
        words_list=[]
        second_tags_list=[]
        index_pairs_list=[]
        sentences=[]
        speakers = []
        for index, tag in enumerate(tags):
            if (tag in triggers and tag.find('B_') != -1):
                '''发现触发词'''
                type = tag[2:]
                # sentence_char_index_pair = words_in_sentence_index_pairs[index]
                tag_index_pair = [index, index]

                #寻找完整的触发词
                for endIndex in range(index + 1, len(tags)):
                    if (tags[endIndex] == 'I_' + type):
                        tag_index_pair[1] += 1
                    else:
                        break
                words_list.append([x for x in words])
                second_tags_list.append(self.__get_final_tag(type,tag_index_pair,tags,sentence))
                index_pairs_list.append([x for x in words_in_sentence_index_pair])
                sentences.append(sentence)
                speakers.append(speaker)
        return words_list, second_tags_list, index_pairs_list, sentences, speakers

    def __get_final_tag(self,type,index_pair,first_tags_list,sentence):
        final_tags = ['O' for _ in first_tags_list]
        self.__label__(final_tags,index_pair[0],index_pair[1],type+'_Trigger')

        negated_index_pair = self.__findFoward__(index_pair[0], first_tags_list, 'Negated')
        if (not negated_index_pair):
            negated_index_pair = self.__findBack__(index_pair[1], first_tags_list, 'Negated')
        if (negated_index_pair):
            self.__label__(final_tags, negated_index_pair[0], negated_index_pair[1], 'Negation')

        if(type=='Know'):
            time_index_pair = self.__findFoward__(index_pair[0],first_tags_list,'Time')
            if(time_index_pair):
                self.__label__(final_tags,time_index_pair[0],time_index_pair[1],type+'_Time')
        elif(type=='BeInLove'):
            time_index_pair = self.__findFoward__(index_pair[0], first_tags_list, 'Time')
            if (time_index_pair):
                self.__label__(final_tags, time_index_pair[0], time_index_pair[1], type + '_Time')
        elif(type=='Marry'):
            time_index_pair = self.__findFoward__(index_pair[0], first_tags_list, 'Time')
            if (time_index_pair):
                self.__label__(final_tags, time_index_pair[0], time_index_pair[1], type + '_Time')
        elif(type=='Remarry'):
            participant_index_pair = self.__findFoward__(index_pair[0], first_tags_list, 'Person')
            if (participant_index_pair):
                self.__label__(final_tags, participant_index_pair[0], participant_index_pair[1], type + '_Participant')
        elif(type=='Bear'):
            dateOfBirth_index_pair = self.__findFoward__(index_pair[0],first_tags_list,'Time')
            if(dateOfBirth_index_pair):
                self.__label__(final_tags,dateOfBirth_index_pair[0],dateOfBirth_index_pair[1],type+'_DateOfBirth')

            gender_index_pair = self.__findBack__(index_pair[1], first_tags_list, 'Gender')
            if (gender_index_pair):
                self.__label__(final_tags, gender_index_pair[0], gender_index_pair[1], type + '_Gender')

            childName_index_pair = self.__findBack__(index_pair[1], first_tags_list, 'Name')
            if (childName_index_pair):
                self.__label__(final_tags, childName_index_pair[0], childName_index_pair[1], type + '_ChildName')

            age_index_pair = self.__findBack__(index_pair[1], first_tags_list, 'Age')
            if (age_index_pair):
                self.__label__(final_tags, age_index_pair[0], age_index_pair[1], type + '_Age')
        elif(type=='DomesticViolence'):
            if(sentence.find('被')!=-1):
                perpetrators = self.__findFoward__(index_pair[0],first_tags_list,'Person',quickStop=True)
                if (perpetrators):
                    self.__label__(final_tags, perpetrators[0], perpetrators[1], type + '_Perpetrators')
                victim = self.__findFoward__(index_pair[0],first_tags_list,'Person')
                if (victim):
                    self.__label__(final_tags, victim[0], victim[1], type + '_Victim')
            else:
                perpetrators = self.__findFoward__(index_pair[0],first_tags_list,'Person')
                if (perpetrators):
                    self.__label__(final_tags, perpetrators[0], perpetrators[1], type + '_Perpetrators')
                victim = self.__findBack__(index_pair[1],first_tags_list,'Person')
                if (victim):
                    self.__label__(final_tags, victim[0], victim[1], type + '_Victim')
            time = self.__findFoward__(index_pair[0], first_tags_list, 'Time')
            if (time):
                self.__label__(final_tags, time[0], time[1], type + '_Time')
        elif(type=='BadHabit'):
            perpetrators = self.__findFoward__(index_pair[0], first_tags_list, 'Person')
            if ( perpetrators):
                self.__label__(final_tags, perpetrators[0], perpetrators[1], type + '_Perpetrators')
        elif(type=='Derailed'):
            time = self.__findFoward__(index_pair[0], first_tags_list, 'Time')
            if ( time):
                self.__label__(final_tags, time[0], time[1], type + '_Time')
            derailer = self.__findFoward__(index_pair[0], first_tags_list, 'Person')
            if( derailer):
                self.__label__(final_tags, derailer[0], derailer[1], type + '_Derailer')
        elif(type=='Separation'):
            beginTime = self.__findFoward__(index_pair[0], first_tags_list, 'Time')
            if( beginTime):
                self.__label__(final_tags, beginTime[0], beginTime[1], type + '_BeginTime')
            endTime = self.__findBack__(index_pair[1],first_tags_list,'Time')
            if( endTime):
                self.__label__(final_tags,endTime[0],endTime[1],type+'_EndTime')
            duration = self.__findBack__(index_pair[1], first_tags_list, 'Duration')
            if ( endTime):
                self.__label__(final_tags, duration[0], duration[1], type + '_Duration')
        elif(type=='DivorceLawsuit'):
            sueTime = self.__findFoward__(index_pair[0], first_tags_list, 'Time')
            if ( sueTime):
                self.__label__(final_tags, sueTime[0], sueTime[1], type + '_SueTime')
            initiator = self.__findFoward__(index_pair[0], first_tags_list, 'Person')
            if ( initiator):
                self.__label__(final_tags, initiator[0], initiator[1], type + '_Initiator')

            court = self.__findFoward__(index_pair[0], first_tags_list, 'Court')
            if ( court):
                court = self.__findBack__(index_pair[1], first_tags_list, 'Court')
            if ( court):
                self.__label__(final_tags, court[0], court[1], type+'_Court')

            result = self.__findBack__(index_pair[1], first_tags_list, 'Judgment')
            if ( result):
                self.__label__(final_tags, result[0], result[1], type + '_Result')
            judgeTime = self.__findBack__(index_pair[1], first_tags_list, 'Time')
            if ( judgeTime):
                self.__label__(final_tags, judgeTime[0], judgeTime[1], type + '_JudgeTime')
        elif(type=='Wealth'):
            value = self.__findFoward__(index_pair[0], first_tags_list, 'Price')
            if ( value):
                self.__label__(final_tags, value[0], value[1], type + '_Value')

            isPersonal = self.__findFoward__(index_pair[0], first_tags_list, 'PersonalProperty')
            if ( isPersonal):
                isPersonal = self.__findBack__(index_pair[1], first_tags_list, 'PersonalProperty')
            if ( isPersonal):
                self.__label__(final_tags, isPersonal[0], isPersonal[1], type + '_IsPersonal')

            isCommon = self.__findFoward__(index_pair[0], first_tags_list, 'CommonProperty')
            if ( isCommon):
                isCommon = self.__findBack__(index_pair[1], first_tags_list, 'CommonProperty')
            if ( isCommon):
                self.__label__(final_tags, isCommon[0], isCommon[1], type + '_IsCommon')

            whose = self.__findBack__(index_pair[1], first_tags_list, 'Person')
            if( whose):
                self.__label__(final_tags, whose[0], whose[1], type + '_Whose')
        elif(type=='Debt'):
            creditor = self.__findBack__(index_pair[1],first_tags_list,'Person')
            if( creditor):
                self.__label__(final_tags,creditor[0],creditor[1],'_Creditor')

            value = self.__findBack__(index_pair[1], first_tags_list, 'Price')
            if ( value):
                self.__label__(final_tags, value[0], value[1], type + '_Value')
        elif(type=='Credit'):
            debtor = self.__findBack__(index_pair[1], first_tags_list, 'Person')
            if ( debtor):
                self.__label__(final_tags, debtor[0], debtor[1], '_Creditor')

            value = self.__findBack__(index_pair[1], first_tags_list, 'Price')
            if ( value):
                self.__label__(final_tags, value[0], value[1], type + '_Value')
        return final_tags

    def __findFoward__ (self,end,tags,target,quickStop=False):
        '''find from 0 to self.trigger_begin_index'''
        '''if quickStop = True ,return when first find target, quickStop=False, return until find last target'''
        index_pair = [0,0]
        haveFound = False
        # 从头找到触发词index
        for index in range(0, end):
            if (tags[index] == 'B_'+target):  # 先找B_ ,后面找到的覆盖前面找到的
                if(haveFound and quickStop):
                    break
                index_pair[0] = index
                index_pair[1] = index
                haveFound=True
                '''找到完整的'''
                for index2 in range(index+1,end):
                    if(tags[index2]=='I_'+target):
                        index_pair[1]=index2
                    else:
                        break
        if(not haveFound):
            return None
        return index_pair

    def __findBack__(self,begin,tags,target):
        index_pair = [0, 0]
        end_index= begin+1
        # 从触发词开始往后找
        hasFound = False
        while (end_index < len(tags)):
            if (tags[end_index] == 'B_' + target):  # 先找B_
                index_pair[0] = end_index
                index_pair[1] = end_index
                hasFound = True
                '''找到完整的'''
                for index2 in range(end_index+1, len(tags)):
                    if (tags[index2] == 'I_' + target):
                        index_pair[1] = index2
                    else:
                        break
                hasFound=True
                break
            end_index+=1
        if(not hasFound):
            return None
        return index_pair

    def __label__(self,tags,begin,end,tag):
        tags[begin] = 'B_'+tag
        for index in range(begin+1,end+1):
            tags[index] = 'I_'+tag

#判断是否含有关注事实触发词
def ifContainTrigger(sentence):
    # 初始化触发词集
    triggerDict = CONFIG.TRIGGER_WORDS_DICT
    # 判断释放含触发词
    triggerContained = ''
    for oneKindTrigger in triggerDict.items():
        triggerType = oneKindTrigger[0]
        for word in oneKindTrigger[1]:
            if (sentence.find(word) != -1):
                triggerContained = triggerContained + (triggerType + ':' + word)
                break
    if (len(triggerContained) == 0):
        return False
    return True
if __name__ == '__main__':
    ext = Extractor()
    events = ext.extractor('原、被告双方1986年上半年经人介绍认识，××××年××月××日在临桂县宛田乡政府登记结婚，××××年××月××日生育女儿李某乙，××××年××月××日生育儿子李某丙，现女儿李某乙、儿子李某丙都已独立生活')
    json_str = events[0]._obj_to_json()
    print('end')
    exit(0)