#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
    处理翻译解码文件，将其划分为翻译结果和参考结果文件.
    Author: Lihui Wang     
    Date: 2019-03-05
''' 

import sys
from sys import argv
import pdb

def divideFile(resultfilename, referencefilename, totalfilename):
    f_result = open(resultfilename, 'w', encoding = 'utf-8')
    f_reference = open(referencefilename, 'w', encoding = 'utf-8')
    with open(totalfilename, 'r', encoding = 'utf-8') as f_in:
        lines = f_in.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('Decoded'):
                line = line.split(' ', 1)[1]
                wordList = line.split()[1 : -1]
                string = ' '.join(wordList)
                f_result.write(string.strip() + '\n')
            elif line.startswith('Targets'):
                line = line.split(' ', 1)[1]
                wordList = line.split()[1 : -1]
                string = ' '.join(wordList)
                f_reference.write(string.strip() + '\n')
    f_result.close()
    f_reference.close()

if __name__ == '__main__':
    argvLen = len(sys.argv)
    if argvLen != 4:
        print('<Usage>: python dealResultReference.py totalFilename resultFilename referenceFilenam.\n')
        exit(1)
    totalFilename = argv[1]
    resultFilename = argv[2]
    referenceFilenam = argv[3]
    divideFile(resultFilename, referenceFilenam, totalFilename)
