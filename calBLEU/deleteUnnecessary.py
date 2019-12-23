#!/usr/bin/python
#encoding=utf-8
'''
    处理生成的汉语译文数据
    Create Date: 2019-05-18
    Author: Lihui Wang
'''
import re
from sys import argv
import sys
import pdb


'''
    读取filename文件内容，正文内容一行存为一个dict条目
    @params     file name       str
    @params     have space      str
    @returns    content dict    dict(ID--content)
'''
def readFile(filename, haveSpace):
    contentDict = dict()
    with open(filename, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
        count = 1
        for line in lines:
            line = line.strip()
            if len(line) != 0:
                content = post_processing(line, haveSpace) + '\n'
                contentDict[count] = content
                count = count + 1
            else:
                continue
    print('There are ' + str(count - 1) + ' lines in file ' + filename)
    return contentDict

'''
    将一个string进行后处理，删除'<eos>'之后的内容，删除'Decoder: <unk>'等标志
    @params     input string       str
    @params     have space         str
    @returns    output string      str
'''
def post_processing(string, haveSpace):
    wordList = string.split()
    result = ""
    for word in wordList:
        if word == 'Decoded:':
            continue
        elif word == '<unk>':
            continue
        elif word == '<eos>':
            break
        else:
            if haveSpace == 'yes':
                result = result + word + ' '
            else:
                result = result + word
    return result

'''
    将contentDict中的每句话(value)写入output文件
    @params     content dict    dict(ID--content)
    @params     file name       str
'''
def writeTxt(contentDict, outputfilename):
    f_out = open(outputfilename, 'w', encoding = 'utf-8')
    count = 0

    for keyid, content in contentDict.items():
        content = content.strip()
        count = count + 1
        f_out.write(content + '\n')
    print('There are ' + str(count) + ' lines writted to the file ' + outputfilename)
    
    f_out.close()
    

if __name__ == '__main__':
    argvLen = len(sys.argv)
    if argvLen != 4:
        print('<Usage>: python deleteUnnecessary.py inputfilename outputfilename haveSpace(yes/no).\n')
        exit(1)
    inputfilename = argv[1]
    outputfilename = argv[2]
    haveSpace = argv[3]

    contentDict = readFile(inputfilename, haveSpace)
    writeTxt(contentDict, outputfilename)

