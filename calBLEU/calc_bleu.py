#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
    计算翻译结果的BLEU值.
    Author: Shaojun Gao, Lihui Wang     
    Date: 2019-03-03
''' 

import nltk
import sys
import pdb

if __name__ == '__main__':
    hypfile = sys.argv[1]
    reffile = sys.argv[2]
    lines1 = open(hypfile, 'r', encoding='utf-8').readlines()
    lines2 = open(reffile, 'r', encoding='utf-8').readlines()
    if len(lines1) != len(lines2):
        print ("lenght do not equal")
        sys.exit(1)
    else:
        total_bleu = 0
        for i in range(len(lines1)):
            hypsen = lines1[i].split()
            refsen = lines2[i].split()

            try:
                bleu = nltk.translate.bleu_score.sentence_bleu([refsen], hypsen, weights=[1])
                #print (bleu)
                total_bleu += bleu
            except:
                print ("blue error")

        average_bleu = total_bleu / len(lines1)
        print('Total BLEU is ' + str(total_bleu))
        print('Average BLEU is ' + str(average_bleu))

