#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'


from model import parties,event_types,Entity,Event,Relation

class single_file_handler(object):
    def __init__(self):
        self.entity_map = None
        self.event_map = None

    def handler(self,file,origin_file):
        '''init gloabl variable'''
        self.event_map = {}
        self.entity_map = {}

        event_lines = []
        relation_lines = []

        '''iterator the file content'''
        with open(file,'r',encoding='utf8') as reader:
            line = reader.readline()
            while(line):
                if(line.startswith('T')):
                    '''if it's entity , handle immediatedly'''
                    entity = self.__handleEntity__(line)
                    self.entity_map[entity.id] = entity
                elif(line.startswith('E')):
                    '''if it's event, must handle after all entity have been handled'''
                    event_lines.append(line)
                elif(line.startswith('R')):
                    '''if it's relation, must handle after all event have been handled'''
                    relation_lines.append(line)
                line = reader.readline()

        '''handle event'''
        for event_line in event_lines:
            event = self.__handleEvent(event_line,origin_file)
            self.event_map[event.id] = event

        '''handle relation'''
        relations = []
        for relation_line in relation_lines:
            relations.append(self.__handleRelation(relation_line))

        return self.event_map,relations

    def __handleEntity__(self,line):
        '''handle entity, for example: T1 Know 16 18 相识'''
        words = line.split()
        entity = Entity(words[0],words[1],words[2],words[3],words[4])
        return entity

    def __handleEvent(self,line,origin_file_path):

        '''handle event, for example: '''
        '''E1 Marry:T3 Time:T2'''

        '''the first argu is trigger:'''
        '''Marry:T3'''

        words = line.split()
        event_id = words[0]

        trigger = words[1]
        type_and_trigger_id = trigger.split(':')
        event_type = type_and_trigger_id[0]
        trigger_id = type_and_trigger_id[1]

        trigger_entity = self.entity_map[trigger_id]

        '''record the begin and end index of the event in the origin file'''
        event_sentence_begin_index = trigger_entity.begin
        event_sentence_end_index = trigger_entity.end

        event = Event(event_id,event_types.get_event_type(event_type),trigger_entity)
        for word in words[2:]:
            entity_type_and_entity_id = word.split(':')
            argu_entity = self.entity_map[entity_type_and_entity_id[1]]

            event_sentence_begin_index = min(event_sentence_begin_index,argu_entity.begin)
            event_sentence_end_index = max(event_sentence_end_index,argu_entity.end)
            '''set the argu of the event'''
            event.__setArgu(entity_type_and_entity_id[0], argu_entity)

        '''find the origin sentence in the origin file'''
        '''find the speaker of the event'''
        with open(origin_file_path,'r',encoding='utf8') as origin_reader:
            cursor = 0

            '''default speaker is plaintiff'''
            speaker = parties.PLAINTIFF
            for line in origin_reader.readlines():
                if (line.find('被告')!=-1 and line.find('辩称')!=-1):
                    speaker = parties.DEFENDANT
                line = line.replace('\n', '\r\n')

                begin_index_of_the_line = cursor
                end_index_of_the_line = cursor + len(line)

                '''smaller than the begin index'''
                if (end_index_of_the_line <= event_sentence_begin_index):
                    cursor = end_index_of_the_line
                    continue

                if (begin_index_of_the_line <= event_sentence_begin_index and event_sentence_begin_index <= end_index_of_the_line
                        and begin_index_of_the_line <= event_sentence_end_index and event_sentence_end_index <= end_index_of_the_line):
                    '''整个事件句子在当前句子中'''
                    event.__addSent(line.strip())
                    event.begin_index = begin_index_of_the_line

                    break

                elif(begin_index_of_the_line <= event_sentence_begin_index and event_sentence_begin_index <= end_index_of_the_line and
                      end_index_of_the_line < event_sentence_end_index):
                    '''只有事件句子的开始在当前句子中'''
                    event.__addSent(line.strip())
                    event.begin_index = begin_index_of_the_line

                elif (event_sentence_begin_index < begin_index_of_the_line and begin_index_of_the_line <= event_sentence_end_index and
                     event_sentence_end_index<= end_index_of_the_line):
                    '''只有事件句子的结尾在当前句子中'''
                    event.__addSent(line)
                    break
                elif(event_sentence_begin_index<=begin_index_of_the_line and end_index_of_the_line<=event_sentence_end_index):
                    '''当前句子整个都在事件句子中'''
                    event.__addSent(line)
                cursor = end_index_of_the_line
            event.sentence = event.sentence.replace('\r\n','').replace('\n','')
            event.speaker = speaker

        return event

    def __handleRelation(self,line):
        '''handle relation, for example:'''
        '''R1 Contradictory Arg1:E6 Arg2:E3'''
        ''' 'Contradictory' can be 'Entailment' '''
        words = line.split()

        arg1_event_id = words[2].split(':')[1]
        arg2_event_id = words[3].split(':')[1]

        relation = Relation(words[0],words[1],self.event_map[arg1_event_id],self.event_map[arg2_event_id])
        return relation


if __name__ == '__main__':
    pass