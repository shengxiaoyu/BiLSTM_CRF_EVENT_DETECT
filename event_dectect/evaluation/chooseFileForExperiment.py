#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'

import os
import shutil

def main():
    baseDir = r'A:\Bi-LSTM+CRF\原始_待分句_样例\cpws'
    targetDir = r'C:\Users\13314\Desktop\test'
    # choose2KB(baseDir,targetDir)
    chooseNumberOfChar(baseDir,targetDir,1500)

def choose2KB(baseDir,targetDir):
    count = 0
    for i in range(10, 100):
        dir = os.path.join(baseDir, str(i))
        if (count == 150):
            break
        for fileName in os.listdir(dir):
            path = os.path.join(dir, fileName)
            if (os.path.getsize(path) > 1.9 * 1024 and os.path.getsize(path) < 2.1 * 1024):
                with open(path, 'r', encoding='utf8') as f:
                    content = f.read()
                    if (content.find('诉称') != -1 and content.find('辩称') != -1):
                        shutil.copy(path, targetDir)
                        count += 1
                        print(path)
                        if (count == 150):
                            break
def chooseNumberOfChar(baseDir,targetDir,numberChars):
    count = 0
    content = ''
    for i in range(10, 100):
        dir = os.path.join(baseDir, str(i))
        if (count == 20):
            break
        for fileName in os.listdir(dir):
            path = os.path.join(dir, fileName)
            with open(path, 'r', encoding='utf8') as f:
                content = f.read()
                if(abs(len(content)-numberChars)<30):
                    if (content.find('诉称') != -1 and content.find('辩称') != -1):
                        shutil.copy(path, targetDir)
                        count += 1
                        print(path)
                        if (count == 20):
                            break

if __name__ == '__main__':
    main()
    # pass