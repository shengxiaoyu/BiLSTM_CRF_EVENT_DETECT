#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'

import os
import docx
# from win32com import client as wc
from event_dectect.data_pre_process.data_process import DataProcess
from enum import Enum
from pyltp import SentenceSplitter

#审判员提问开头：
SPY = set(['？','审判员','问','审'])
#原告答
YG = Enum('YG',('原','原告','答'))
#被告答
BG = Enum('BG',('被','被告','答'))
#unknown
UNKNOWN = Enum('UNKNOWN',('答','均答'))

class TsblProcess(DataProcess):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        # self.docTool = wc.Dispatch('Word.Application')

    def segmentAndSave(self,segmentor,segment_result_save_path,mode):
        for ah in os.listdir(self.root_dir):#单个案件
            #获取庭审调查内容段
            tsdc = self.__getTsdc__(os.path.join(self.root_dir,ah))

            if(tsdc == None or len(tsdc)==0):
                print(ah + '空')
                continue

            with open(os.path.join(segment_result_save_path,ah+'.txt'),'a',encoding='utf8') as fw:
                fw.write('\n庭审调查：\n')
                for paragraph in tsdc:
                    if(mode == 'sentence'):
                    # 分句
                        sentences = SentenceSplitter.split(paragraph)
                        if(len(sentences)>0):
                            for sentence in sentences :
                                fw.write(sentence.strip())
                                fw.write('\n')
                    elif(mode=='paragraph'):
                        paragraph = paragraph.strip()
                        if(len(paragraph)>0):
                            fw.write(paragraph+'\n')

    def __getTsdc__(self,path):#获取庭审调查内容
        ajTsbl = os.path.join(path, '庭审笔录')
        beginIndex = None
        isBegin = False
        result = [] #最终返回结果

        for file in os.listdir(ajTsbl):  # 案号//庭审笔录//单个庭审笔录文件
            if(file.startswith('~$')): #打开文件时出现的一些奇怪文件
                continue
            theTsdc = []
            content = None
            #获取文件全部内容
            filePath = os.path.join(ajTsbl,file)
            if (file.endswith(r'.doc')):
                content = self.__getDocContent__(filePath)
            elif (file.endswith(r'.docx')):
                content = self.__getDocxContent__(filePath)
            else:
                # print('文件类型不是doc，docx忽略：' + filePath)
                continue

            if (content == None):
                continue

            #获取庭审调查内容
            for index,paragragh in enumerate(content):
                if(paragragh==''):
                    continue
                #庭审调查内容最先有
                # 1、“法庭调查”，或者“继续开庭”
                # 2、“事实与理由”，“事实和理由”，“事实理由”，“事实及理由”，有可能紧接下一段就是法庭调查内容，也有可能紧接几段都是事实理由内容
                # 3、一般含“诉状”的下一段是法庭调查内容
                # 4、被告答辩开始就是法庭调查内容

                #开始法庭调查
                if(paragragh.find('法庭调查')!=-1
                        and paragragh.find('结束')==-1):
                    isBegin = True
                    beginIndex = index+1
                    continue

                if (paragragh.find('继续开庭') != -1):
                    isBegin = True
                    beginIndex = index
                    theTsdc = []

                if(paragragh.find('事实及理由')!=-1
                        or paragragh.find('事实与理由')!=-1
                        or paragragh.find('事实和理由')!=-1
                        or paragragh.find('事实理由')!=-1):
                    #查看后面哪一段的开头是审判员提问
                    for theIndex in range(index+1,len(content)):
                        if(self.__isBeginWithSPY__(content[theIndex])):
                            beginIndex = theIndex
                            isBegin = True
                            theTsdc = []
                            break

                    #如果没找到就默认取下一段开始
                    if(not isBegin):
                        beginIndex = theIndex + 1
                        isBegin = True
                        continue

                #一般来说，此时从下一段开始作为庭审调查内容
                if(paragragh.find('详见书面补充意见详见书面补充意见')!=-1
                        or paragragh.find('宣读起诉状')!=-1
                        or paragragh.find('详见诉状')!=-1
                        or paragragh.find('同诉状')!=-1
                         or paragragh.find('宣读诉状')!=-1):
                    beginIndex = index+1
                    isBegin = True
                    theTsdc = []
                    continue

                #从此段开始
                if(paragragh.find('被告')!=-1
                        and paragragh.find('答辩')!=-1):
                    beginIndex = index
                    isBegin = True


                #结尾
                if(
                        # paragragh.find('原告举证')!=-1
                        # or (paragragh.find('原告')!=-1 and paragragh.find('证据提交')!=-1)
                        # or (paragragh.find('当事人')!=-1 and paragragh.find('举证')!=-1)
                        # or (paragragh.find('围绕焦点')and paragragh.find('举证')!=-1)
                        # or
                        (paragragh.find('法庭调查')!= -1and paragragh.find('结束')!=-1)
                        or paragragh.find('庭审到此')!=-1
                        or paragragh.find('辩论结束')!=-1
                        or paragragh.find('本院主持')!=-1
                        # or paragragh.find('向法庭提交证据')!=-1
                        # or paragragh.find('有何证据提交法庭')!=-1
                        or paragragh.find('休庭')!=-1
                        or paragragh.find('闭庭')!=-1
                        or paragragh.find('今天庭审到此')!=-1
                        or paragragh.find('今天开庭到此')!=-1
                        or paragragh.find('审判员制作并宣读调解书')!=-1):
                    isBegin = False
                    beginIndex = None
                    break
                if(isBegin and index>=beginIndex):
                   theTsdc.append(paragragh)

            result.extend(theTsdc)
            theTsdc = []

        return result

    ##是否是审判员提问开头
    def __isBeginWithSPY__(self,str):
        for var in SPY:
            if(str.startswith(var)):
                return True
        return False

    ##是否是原被告回答开头


    ###获取Doc内容
    def __getDocContent__(self,path):

        if(not path.endswith('.doc')):
            return
        #需要先转换为docx
        newPath = path.replace(r'.doc',r'.docx')
        if os.path.exists(newPath):#已经存在，则说明转换过了，在docx列表中会读
            # print('存在同名docx文件，可通过读取该文件获得庭审信息:'+newPath)
            return
        try:
            doc = self.docTool.Documents.Open(path)
            doc.SaveAs(newPath, 12)
            doc.Close()
        except:
            print('!!!!!读取文件发生错误：'+path)
            return  None
        content =  self.__getDocxContent__(newPath)
        os.remove(newPath)
        return content


    ##获取Docx内容
    def __getDocxContent__(self,path):
        if(not path.endswith('.docx')):
            return
        file = docx.Document(path)
        content = list(filter(lambda x:x!='',map(lambda x:x.text.strip().replace(' ',''),file.paragraphs)))
        return content

    def __colse__(self):
        self.docTool.Quit()



if __name__ == '__main__':
    rootdir = r'E:\研二1\学术论文\准备材料2\离婚纠纷第二批（分庭审笔录）\含庭审笔录'

    pass