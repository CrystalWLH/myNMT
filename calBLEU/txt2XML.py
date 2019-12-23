#!/usr/bin/python
#encoding=utf-8
'''
    处理翻译结果，将txt转为xml
    Create Date: 2019-11-04
    Author: Lihui Wang
'''
import re
from sys import argv
import sys
import pdb

'''
    读取filename文件内容，正文内容一行存为一个dict条目 （中文）
    @params     file name       str
    @params     have space      str
    @returns    content dict    dict(ID--content)
'''
def readFileZh(filename, haveSpace):
    contentDict = dict()
    with open(filename, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
        count = 1
        for line in lines:
            line = line.strip()
            if len(line) != 0:
                #content = post_processing(line, haveSpace) + '\n'
                #contentDict[count] = content
                if haveSpace == 'have':
                    contentDict[count] = line + '\n'
                else:
                    string = ''.join(line.split(' ')) + '\n'
                    contentDict[count] = string
                count = count + 1
            else:
                contentDict[count] = '\n'
                count = count + 1
    print('There are ' + str(count - 1) + ' lines in file ' + filename)
    return contentDict

'''
    读取filename文件内容，正文内容一行存为一个dict条目 （英文）
    @params     file name       str
    @returns    content dict    dict(ID--content)
'''
def readFileEn(filename):
    contentDict = dict()
    with open(filename, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
        count = 1
        for line in lines:
            line = line.strip()
            if len(line) != 0:
                #content = post_processing(line, haveSpace) + '\n'
                #contentDict[count] = content
                string = ' '.join(line.split(' ')) + '\n'
                contentDict[count] = string
                count = count + 1
            else:
                contentDict[count] = '\n'
                count = count + 1
    print('There are ' + str(count - 1) + ' lines in file ' + filename)
    return contentDict


'''
    将contentDict中的每句话(value)按照xml文件的格式写入output文件 （藏汉翻译）
    @params     content dict    dict(ID--content)
    @params     file name       str
'''
def writeXMLFileTiZh(contentDict, outputfilename):
    f_out = open(outputfilename, 'w', encoding = 'utf-8')
    count = 0
    
    string = '<?xml version="1.0" encoding="UTF-8"?>' + '\n'
    f_out.write(string)
    string = '<tstset setid="ti_zh_gove_trans" srclang="ti" trglang="zh">' + '\n'
    f_out.write(string)
    string = '<system site="PKU_SHRC" sysid="CCMT2019_012">' + '\n'
    f_out.write(string)
    string = '软硬件环境：Linux vision-server 4.15.0-43-generic #46~16.04.1-Ubuntu' + '\n'
    f_out.write(string)
    string = '</system>' + '\n'
    f_out.write(string)
    string = '<DOC docid="gove" sysid="CCMT2019_012">' + '\n'
    f_out.write(string)
    string = '<p>' + '\n'
    f_out.write(string)

    for keyid, content in contentDict.items():
        content = '<seg id="' + str(keyid) + '">' + content.strip() + '</seg>'
        count = count + 1
        f_out.write(content + '\n')
    print('There are ' + str(count) + ' lines writted to the file ' + outputfilename)
    
    string = '</p>' + '\n'
    f_out.write(string)
    string = '</DOC>' + '\n'
    f_out.write(string)
    string = '</tstset>' + '\n'
    f_out.write(string)
    f_out.close()

'''
    将contentDict中的每句话(value)按照xml文件的格式写入output文件 （中英翻译）
    @params     content dict    dict(ID--content)
    @params     file name       str
'''
def writeXMLFileZhEn(contentDict, outputfilename):
    f_out = open(outputfilename, 'w', encoding = 'utf-8')
    count = 0
    
    string = '<?xml version="1.0" encoding="UTF-8"?>' + '\n'
    f_out.write(string)
    string = '<tstset setid="zh_en_trans" srclang="zh" trglang="en">' + '\n'
    f_out.write(string)
    string = '<system site="PKU_SHRC" sysid="zh_en_sys">' + '\n'
    f_out.write(string)
    string = '软硬件环境：Linux vision-server 4.15.0-43-generic #46~16.04.1-Ubuntu' + '\n'
    f_out.write(string)
    string = '</system>' + '\n'
    f_out.write(string)
    string = '<DOC docid="AI_chanllenger" sysid="zh_en_sys">' + '\n'
    f_out.write(string)
    string = '<p>' + '\n'
    f_out.write(string)

    for keyid, content in contentDict.items():
        content = '<seg id="' + str(keyid) + '">' + content.strip() + '</seg>'
        count = count + 1
        f_out.write(content + '\n')
    print('There are ' + str(count) + ' lines writted to the file ' + outputfilename)
    
    string = '</p>' + '\n'
    f_out.write(string)
    string = '</DOC>' + '\n'
    f_out.write(string)
    string = '</tstset>' + '\n'
    f_out.write(string)
    f_out.close()


if __name__ == '__main__':
    argvLen = len(sys.argv)
    if argvLen != 5:
        print('<Usage>: python txt2xml.py inputfilename outputfilename language(zh/en) haveSpace(have/not).\n')
        exit(1)
    inputfilename = argv[1]
    outputfilename = argv[2]
    language = argv[3]
    haveSpace = argv[4]

    #pdb.set_trace()
    if language == 'zh':
        contentDict = readFileZh(inputfilename, haveSpace)
        writeXMLFileTiZh(contentDict, outputfilename)
    elif language == 'en':
        contentDict = readFileEn(inputfilename)
        writeXMLFileZhEn(contentDict, outputfilename)
    else:
        print('Please input correct language(zh/en).')
        exit(1)

