#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pyltp import SentenceSplitter

__doc__ = 'description'
__author__ = '13314409603@163.com'

import xml.sax
from data_pre_process.data_process import DataProcess
import os
from pyltp import SentenceSplitter

class CpwsProcess(DataProcess):
    def __init__(self,rootDir):
        self.rootDir = rootDir
        self.parser = xml.sax.make_parser()
        self.parser.setFeature(xml.sax.handler.feature_namespaces, 0)
        self.handler = CpwsHandler()
        self.parser.setContentHandler(self.handler)

    def segmentAndSave(self,segmentor,segment_result_save_path):

        def segmentParasgraphAndSave(parasgraph,fileWriter):
            sentences = SentenceSplitter.split(parasgraph)
            for sentence in sentences:
                sentence = sentence.strip()
                if(sentence==None or len(sentence)==0):
                    continue
                fileWriter.write(sentence)
                fileWriter.write('\n')
                # words = segmentor.segment(sentence)
                # # 去除标点符号
                # words = self.delBdfh(words)
                # if (len(words) == 0):
                #     continue
                # fileWriter.write(' '.join(words))
                # fileWriter.write('\n')


        for fileName in os.listdir(self.rootDir):
            self.parser.parse(os.path.join(self.rootDir,fileName))
            content = self.handler.getContentAndClear()
            # 分句
            if(content!=None):
                savePath = os.path.join(segment_result_save_path,fileName)
                # savePath = savePath.replace('（','(').replace('）',')')
                with open(savePath,'w',encoding='utf8') as f:
                    segmentParasgraphAndSave(content.get('YGSCD'),f)
                    f.write('\n')
                    segmentParasgraphAndSave(content.get('BGBCD'),f)



class CpwsHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.AH = None
        self.YGSCD = None
        self.BGBCD = None

    def startElement(self,tag,attributes):
        if(tag == 'AH'):
            self.AH = attributes['value']
        elif(tag == 'YGSCD'):
            self.YGSCD = attributes['value']
        elif(tag == 'BGBCD'):
            bc = attributes['value']
            if (bc != None and len(bc) != 0 and (bc.find('被告') != -1 and bc.find('辩称') != -1)):
                self.BGBCD = bc
    def endElement(self,tag):
        pass
    def characters(self,content):
        pass
    def getContentAndClear(self):
        content = None
        if(self.AH!=None and self.YGSCD!=None and self.BGBCD!=None):
            content =  {'AH':self.AH,
                'YGSCD':self.YGSCD,
                'BGBCD':self.BGBCD}
        self.BGBCD = None
        self.YGSCD = None
        self.AH = None
        return content

if __name__ == '__main__':

    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces,0)
    handler = CpwsHandler()
    parser.setContentHandler(handler)
    parser.parse('E:\\研二2\\2013\\116.xml')
    print(handler.getContentAndClear())
    pass