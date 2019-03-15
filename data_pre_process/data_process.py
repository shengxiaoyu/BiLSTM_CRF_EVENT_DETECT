__doc__ = 'description'
__author__ = '13314409603@163.com'

import os
import sys

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

if __name__ == '__main__':
    # an2Index('C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\segment_result\\qstsbl')
    an2Index('C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\segment_result\\qsz')
    sys.exit(0)