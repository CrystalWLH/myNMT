#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
    CTC实现解码计算准确率的文件
    Author: Lihui Wang     
    Create Date: 2019-02-27
''' 

import os
import math
import torch
import pdb
import numpy as np

'''
    解码器基类定义，作用是将模型的输出转化为文本使其能够与标签计算正确率
'''
class Decoder(object):
    def __init__(self, int2char, space_idx = 1, blank_index = 0):
        '''
        int2char     :     将类别转化为字符标签
        space_idx    :     空格符号的索引，如果为为-1，表示空格不是一个类别
        blank_index  :     空白类的索引，默认设置为0
        '''
        self.int_to_char = int2char
        self.space_idx = space_idx
        self.blank_index = blank_index
        self.num_word = 0
        self.num_char = 0

    #解码函数，在GreedyDecoder和BeamDecoder继承类中实现
    def decode(self):
        raise NotImplementedError;

    def phone_word_error(self, prob_tensor, frame_seq_len, targets, target_sizes):
        '''计算词错率和字符错误率
        Args:
            prob_tensor     :   模型的输出
            frame_seq_len   :   每个样本的帧长
            targets         :   样本标签
            target_sizes    :   每个样本标签的长度
        Returns:
            wer             :   词错率，以space为间隔分开作为词
            cer             :   字符错误率
        '''
        strings = self.decode(prob_tensor, frame_seq_len)
        targets = self._unflatten_targets(targets, target_sizes)
        target_strings = self._process_strings(self._convert_to_strings(targets))
        
        cer = 0
        wer = 0
        for x in range(len(target_strings)):
            cer += self.cer(strings[x], target_strings[x])
            wer += self.wer(strings[x], target_strings[x])
            self.num_word += len(target_strings[x].split())
            self.num_char += len(target_strings[x])
        return cer, wer

    def _unflatten_targets(self, targets, target_sizes):
        '''将标签按照每个样本的标签长度进行分割
        Args:
            targets        :    数字表示的标签
            target_sizes   :    每个样本标签的长度
        Returns:
            split_targets  :    得到的分割后的标签
        '''
        split_targets = []
        for i in range(len(targets)):
            split_targets.append(targets[i][ : target_sizes[i]])
        return split_targets

    def _process_strings(self, seqs, remove_rep = False): 
        '''处理转化后的字符序列，包括去重复等，将list转化为string
        Args:
            seqs       :   待处理序列
            remove_rep :   是否去重复
        Returns:
            processed_strings  :  处理后的字符序列
        '''
        processed_strings = []
        for seq in seqs:
            string = self._process_string(seq, remove_rep)
            processed_strings.append(string)
        return processed_strings
   
    def _process_string(self, seq, remove_rep = False):
        string = ''
        for i, char in enumerate(seq):
            if char != '@@@@@':
                if remove_rep and i != 0 and char == seq[i - 1]: #remove dumplicates
                    pass
                elif self.space_idx == -1:
                    string = string + ' '+ char
                elif char == self.int_to_char.vocab.itos[self.space_idx]:
                    string += ' '
                else:
                    string = string + char
        return string

    def _convert_to_strings(self, seq, sizes=None):
        '''将数字序列的输出转化为字符序列
        Args:
            seqs       :   待转化序列
            sizes      :   每个样本序列的长度
        Returns:
            strings  :  转化后的字符序列
        '''
        strings = []
        for x in range(len(seq)):
            seq_len = sizes[x] if sizes is not None else len(seq[x])
            string = self._convert_to_string(seq[x], seq_len)
            strings.append(string)
        return strings

    def _convert_to_string(self, seq, sizes):
        result = []
        for i in range(sizes):
            if seq[i] == 50004:
                result.append('@@@@@')
            else:
                result.append(self.int_to_char.vocab.itos[seq[i]])
        if self.space_idx == -1:
            return result
        else:
            return ''.join(result)
    
    '''
        计算两句话的编辑距离，求得分词正确的个数
    '''
    def calEditDistanceCorrect(self, list1, list2):
        m = len(list1)
        n = len(list2)

        # dp = np.zeros((m + 1, n + 1), dtype=np.int)
        dp = [[0 for i in range(n + 1)] for i in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = 0
        for i in range(n + 1):
            dp[0][i] = 0

        for i in range(1 , m + 1):
            for j in range(1 , n + 1):
                if list1[i - 1] == list2[j - 1]:
                    #print(list1[i - 1])
                    temp = max(dp[i][j - 1], dp[i - 1][j])
                    dp[i][j] = max(dp[i - 1][j - 1] + 1, temp)
                else:
                    temp = max(dp[i - 1][j], dp[i][j - 1])
                    dp[i][j] = max(temp, dp[i - 1][j - 1])

        return dp[m][n]

    #计算一个batch的正确词语的个数，分词结果的总词数和ground truth的总词数
    def calWordNumber(self, decodes, labels):
        result_total_word = 0
        truth_total_word = 0
        total_correct = 0
        for x in range(len(labels)):
            result_total_word = result_total_word + len(decodes[x].strip().split()) - 2
            truth_total_word = truth_total_word + len(labels[x].strip().split()) - 2
            resultList = decodes[x].strip().split()[1:-1]
            answerList = labels[x].strip().split()[1:-1]
            total_correct =  total_correct + self.calEditDistanceCorrect(resultList, answerList)

        return result_total_word, truth_total_word, total_correct


'''
    直接解码，把每一帧的输出概率最大的值作为输出值，而不是整个序列概率最大的值
'''
class GreedyDecoder(Decoder):
    def decode(self, prob_tensor, frame_seq_len):
        '''解码函数
        Args:
            prob_tensor   :   网络模型输出
            frame_seq_len :   每一样本的帧数
        Returns:
            解码得到的string，即识别结果
        '''
        prob_tensor = prob_tensor.transpose(0,1)
        _, decoded = torch.max(prob_tensor, 2)
        decoded = decoded.view(decoded.size(0), decoded.size(1))
        decoded = self._convert_to_strings(decoded, frame_seq_len)
        return self._process_strings(decoded, remove_rep=True)

'''
    实现NMT翻译结果的解码
'''
class NMTDecoder(Decoder):
    def decode(self, prob_tensor, trg_len):
        '''解码函数
        Args:
            prob_tensor   :   网络模型输出
            trg_len       :   每一样本的输出长度
        Returns:
            解码得到的string，即识别结果
        '''
        prob_tensor = prob_tensor.transpose(0,1)
        _, decoded = torch.max(prob_tensor, 2)
        decoded = decoded.view(decoded.size(0), decoded.size(1))
        decoded = self._convert_to_strings(decoded, trg_len)
        return self._process_strings(decoded, remove_rep=True)


    def _process_strings(self, seqs, remove_rep = False): 
        '''处理转化后的字符序列，包括去重复等，将list转化为string
        Args:
            seqs       :   待处理序列
            remove_rep :   是否去重复
        Returns:
            processed_strings  :  处理后的字符序列
        '''
        processed_strings = []
        for seq in seqs:
            string = self._process_string(seq, remove_rep)
            processed_strings.append(string)
        return processed_strings
   
    def _process_string(self, seq, remove_rep = False):
        string = ''
        for i, char in enumerate(seq):
            if self.space_idx == -1:
                string = string + ' '+ char
            elif char == self.int_to_char.vocab.itos[self.space_idx]:
                string += ' '
            else:
                string = string + char
        return string
