#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'

class EditDistanceSimilarCalculator(object):
    def cal(self,word1,word2):
        '''1-distance/max_length'''
        return 1-self.editDistance(word1,word2)/max(len(word1),len(word2))

    def editDistance(self,word1,word2):
        if (word2 == None and word1 == None):
            return 0
        if (word1 == None or len(word1) == 0):
            return len(word2)
        if (word2 == None or len(word2) == 0):
            return len(word1)

        tube = [[0 for j in range(len(word2) + 1)] for i in range(len(word1) + 1)]

        for i in range(0, len(word1) + 1):
            for j in range(0, len(word2) + 1):
                if(i==0):
                    tube[i][j] = j
                    continue
                if(j==0):
                    tube[i][j]=i
                    continue
                if (word1[i-1] == word2[j-1]):
                    d = 0
                else:
                    d = 1
                tube[i][j] = min(tube[i][j-1] + 1, tube[i - 1][j] + 1, tube[i - 1][j - 1] + d)
        return tube[len(word1)][len(word2)]
if __name__ == '__main__':
    cal = EditDistanceSimilarCalculator()
    print(cal.editDistance('吵架','争吵'))
    print(cal.cal('吵架','争吵'))