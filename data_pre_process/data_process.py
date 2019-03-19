__doc__ = 'description'
__author__ = '13314409603@163.com'

import os
import sys
import math

BDFH = {'、', ',', '.', '，', '。', '!', '！', '?', '？', ';', '；', ':', '：', '-', '—', '(', '（', ')', '）', '《', '》',
                '“', '”'}
class DataProcess(object):
    # 标点符号集
    def getContent(self):
        raise NotImplementedError()

    def segmentAndSave(self):
        raise NotImplementedError()

    def delBdfh(self,words):
        newWords = []
        for word in words:
            if (word not in BDFH):
                newWords.append(word)
        return newWords
#将案号转为indxe 文件名称
COUNT= 1
def an2Index(path):
    if(not os.path.exists(path)):
        return
    elif(os.path.isdir(path)):
        for file in os.listdir(path):
            an2Index(os.path.join(path,file))
    else:
        global COUNT
        os.renames(path,os.path.join(os.path.dirname(path),str(COUNT)+'.txt'))
        COUNT += 1
#将一个文件夹下的文件列表分割转入子文件夹
def divideFile(savePath,cap):
    fileNames = os.listdir(savePath)
    num = math.ceil(float(len(fileNames))/cap)
    for i in range(num):
        newDir = os.path.join(savePath,str(i+1))
        os.mkdir(newDir)
        beginIndex = i*cap
        endIndex = (i+1)*cap
        endIndex = endIndex if endIndex<len(fileNames) else -1
        for fileName in fileNames[beginIndex:endIndex]:
            os.rename(os.path.join(savePath,fileName),os.path.join(newDir,fileName))

def divideFile2(savePath,num,cap):
    fileNames = filter(lambda x:False if(x.find('.')==-1) else True,os.listdir(savePath))
    fileNames = map(lambda x:x[0:-4],fileNames)
    fileNames = set(fileNames)

    for i in range(num):
        newDir = os.path.join(savePath,str(1+i))
        os.mkdir(newDir)
        minIndex = i*cap
        maxIndex = (i+1)*cap
        for index, fileName in enumerate(fileNames):
            if(index>=minIndex and index<maxIndex):
                os.rename(os.path.join(savePath, fileName + '.ann'),os.path.join(newDir, fileName + '.ann'))
                os.rename(os.path.join(savePath,fileName+'.txt'),os.path.join(newDir,fileName+'.txt'))





if __name__ == '__main__':
    # an2Index('C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\segment_result\\qstsbl')
    # an2Index('C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\segment_result\\qsz')
    divideFile2('C:\\Users\\13314\\Desktop\\test',2,2)
    sys.exit(0)