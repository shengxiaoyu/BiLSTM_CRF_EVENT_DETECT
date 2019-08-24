#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

__doc__ = 'description'
__author__ = '13314409603@163.com'
import csv
def generate(path):
    with open(path,'r',encoding='utf8') as reader, open(path.replace('txt','csv'),'w', newline='',encoding='utf8') as fwriter:
        csv_writer = csv.writer(fwriter,delimiter='\t')
        for line in reader.readlines():
            line = line.strip()
            csv_writer.writerow([line,line,'1'])



if __name__ == '__main__':
    generate(r'A:\研二2\person.txt')
    print('end')
    sys.exit(0)
