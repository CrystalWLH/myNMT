#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
    训练，测试主文件
    Author: Lihui Wang & Shaojun Gao    
    Create Date: 2019-02-25
    Update Date: 2019-12-20
''' 

import os
import math
import argparse
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm, clip_grad_norm_
from torch.nn import functional as F
from model import Encoder, Decoder, Seq2Seq, CTC_Seg, Encoder_Combine
from utils import load_dataset
import pdb
from ctcDecoder import GreedyDecoder, NMTDecoder
import sys
from sys import argv

def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-mode', type=str, default='ctc',
                   help='training mode')
    p.add_argument('-istest', action='store_true',
                   help='train or test.')
    p.add_argument('-epochs', type=int, default=100,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=32,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='in case of gradient explosion')
    p.add_argument('-path', type=str, default='data/',
                   help='the path of data.')
    p.add_argument('-extension', type=str, default='.zh .en',
                   help='the extension of data filename.')
    p.add_argument('-train', type=str, default='train',
                   help='the filename of train set.')
    p.add_argument('-valid', type=str, default='val',
                   help='the filename of valid set.')
    p.add_argument('-test', type=str, default='test',
                   help='the filename of test set.')
    p.add_argument('-src_dict_maxSize', type=int, default=-1,
                   help='max size of src dictionary.')
    p.add_argument('-seg_dict_maxSize', type=int, default=-1,
                   help='max size of seg dictionary.')
    p.add_argument('-tgt_dict_maxSize', type=int, default=-1,
                   help='max size of tgt dictionary.')
    p.add_argument('-model_path', type=str, default=None,
                   help='the path of trained model.')
    p.add_argument('-init_CTC_model', type=str, default=None,
                   help='the model path used by initialize the CTC segmentation network on the combine training.')
    p.add_argument('-init_combine_model', type=str, default=None,
                   help='the model path used by refine the CTC segmentation network on the ctc segmentation training.')
    p.add_argument('-log_FileName', type=str, default='log',
                   help='the file name of log.')
    p.add_argument('-seg_result_file', type=str, default='checkpoints/test_seg_result.txt',
                   help='the path of segmentation test result.')
    p.add_argument('-val_seg_result_file', type=str, default='checkpoints/val_seg_result.txt',
                   help='the path of segmentation val result.')
    p.add_argument('-nmt_result_file', type=str, default='checkpoints/test_nmt_result.txt',
                   help='the path of neural machine translation test result.')
    p.add_argument('-val_nmt_result_file', type=str, default='checkpoints/val_nmt_result.txt',
                   help='the path of neural machine translation val result.')
    p.add_argument('-save_path', type=str, default='checkpoints',
                   help='the path of training model.')
    p.add_argument('-embed_size', type=int, default=256,
                   help='embedding size.')
    p.add_argument('-hidden_size', type=int, default=512,
                   help='hidden size of every layer.')
    p.add_argument('-nmt_encoder_layers', type=int, default=2,
                   help='the number of nmt encoder layers.')
    p.add_argument('-nmt_decoder_layers', type=int, default=2,
                   help='the number of nmt decoder layers.')
    p.add_argument('-ctc_layers', type=int, default=2,
                   help='the number of ctc network layers.')
    p.add_argument('-dropout', type=float, default=0.5,
                   help='dropout.')
    p.add_argument('-model_cell', type=str, default='GRU',
                   help='the cell name of model (LSTM / GRU).')
    return p.parse_args()

def evaluate_ctc(epoch, model, val_iter, vocab_size, ZH_WORD, seg_result_filename = None):
    if seg_result_filename == None:
        filename = 'val_seg_result_' + str(epoch) + '.txt'
        f_out = open(filename, 'w', encoding='utf-8')
    elif epoch == 0:
        f_out = open(seg_result_filename, 'w', encoding = 'utf-8')
    else:
        filename = seg_result_filename.split('.')[0] + '_' + str(epoch) + '.' + seg_result_filename.split('.')[1]
        f_out = open(filename, 'w', encoding='utf-8')

    model.eval()
    total_loss = 0
    result_total_word = 0
    truth_total_word = 0
    total_correct = 0
    ctc_loss = torch.nn.CTCLoss(blank=vocab_size, reduction='mean')
    lenTotal = len(val_iter)
    for b, batch in enumerate(val_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        with torch.no_grad():
            src = Variable(src.data.cuda())
            trg = Variable(trg.data.cuda())
        
        output, hidden = model(src, trg)

        log_probs = output.log_softmax(2)
        trg = torch.transpose(trg, 0, 1)
        temp = torch.ones(list(trg.size())[0], list(trg.size())[1])
        temp = temp.long().cuda()
        #trg = trg + temp
        loss = ctc_loss(log_probs, trg, len_src, len_trg)
        if loss != loss or loss > 1000:
            print('---------Evaluate Loss is big.')
            lenTotal = lenTotal - len(batch)
            continue

        total_loss += loss.data

        #pdb.set_trace()
        #ctc解码获取分词结果
        decoder = GreedyDecoder(ZH_WORD, space_idx=-1, blank_index=vocab_size)

        decoded = decoder.decode(log_probs, len_src)
        targets = decoder._unflatten_targets(trg, len_trg)
        labels = decoder._process_strings(decoder._convert_to_strings(targets))

        f_out.write('Decoded: ' + ' '.join(decoded) + '\n')
        f_out.write('Targets: ' + ' '.join(labels) + '\n')
        f_out.flush()

        #pdb.set_trace()
        result_word, truth_word, correct = decoder.calWordNumber(decoded, labels)
        result_total_word = result_total_word + result_word
        truth_total_word = truth_total_word + truth_word
        total_correct = total_correct + correct

    f_out.close()
    #pdb.set_trace()

    precision = total_correct / result_total_word
    recall = total_correct / truth_total_word
    f_measure = 2 * precision * recall / (precision + recall)
    return total_loss / lenTotal, precision, recall, f_measure

def evaluate_nmt(epoch, model, val_iter, vocab_size, ZH_WORD, EN_WORD, nmt_result_filename = None):
    if nmt_result_filename == None:
        filename = 'val_nmt_result_' + str(epoch) + '.txt'
        f_out = open(filename, 'w', encoding='utf-8')
    elif epoch == 0:
        f_out = open(nmt_result_filename, 'w', encoding = 'utf-8')
    else:
        filename = nmt_result_filename.split('.')[0] + '_' + str(epoch) + '.' + nmt_result_filename.split('.')[1]
        f_out = open(filename, 'w', encoding='utf-8')

    model.eval()
    pad = EN_WORD.vocab.stoi['<pad>']
    total_loss = 0
    for b, batch in enumerate(val_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        with torch.no_grad():
            src = Variable(src.data.cuda())
            trg = Variable(trg.data.cuda())
        output = model(src, trg, teacher_forcing_ratio=0.0)
        loss = F.nll_loss(output[1:].view(-1, vocab_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=pad)
        total_loss += loss.data

        #nmt解码获取翻译结果
        decoder = NMTDecoder(EN_WORD, space_idx=-1, blank_index=vocab_size)

        decoded = decoder.decode(output, len_trg)
        trg = torch.transpose(trg, 0, 1)
        targets = decoder._unflatten_targets(trg, len_trg)
        labels = decoder._process_strings(decoder._convert_to_strings(targets))

        for i in range(len(decoded)):
            f_out.write('Decoded: ' + str(decoded[i]) + '\n')
            f_out.write('Targets: ' + str(labels[i]) + '\n')

        f_out.flush()

    f_out.close()
    return total_loss / len(val_iter)

def evaluate_combine(epoch, model, val_iter, vocab_size, ZH_CHA, EN_WORD, nmt_result_filename = None):
    if nmt_result_filename == None:
        filename = 'val_nmt_result_' + str(epoch) + '.txt'
        f_out = open(filename, 'w', encoding='utf-8')
    elif epoch == 0:
        f_out = open(nmt_result_filename, 'w', encoding = 'utf-8')
    else:
        filename = nmt_result_filename.split('.')[0] + '_' + str(epoch) + '.' + nmt_result_filename.split('.')[1]
        f_out = open(filename, 'w', encoding='utf-8')

    model.eval()
    pad = EN_WORD.vocab.stoi['<pad>']
    total_loss = 0
    for b, batch in enumerate(val_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        with torch.no_grad():
            src = Variable(src.data.cuda())
            trg = Variable(trg.data.cuda())
        output, output_seg = model(src, trg, teacher_forcing_ratio=0.0)
        loss = F.nll_loss(output[1:].view(-1, vocab_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=pad)
        total_loss += loss.data

        #nmt解码获取翻译结果
        decoder = NMTDecoder(EN_WORD, space_idx=-1, blank_index=vocab_size)

        decoded = decoder.decode(output, len_trg)
        trg = torch.transpose(trg, 0, 1)
        targets = decoder._unflatten_targets(trg, len_trg)
        labels = decoder._process_strings(decoder._convert_to_strings(targets))

        for i in range(len(decoded)):
            f_out.write('Decoded: ' + str(decoded[i]) + '\n')
            f_out.write('Targets: ' + str(labels[i]) + '\n')

        f_out.flush()

    f_out.close()
    return total_loss / len(val_iter)

def evaluate_update_twoLoss(epoch, model, val_iter, zh_word_size, en_word_size, ZH_CHA, ZH_WORD, EN_WORD, seg_result_filename = None, nmt_result_filename = None):
    if nmt_result_filename == None:
        filename = 'val_nmt_result_' + str(epoch) + '.txt'
        f_out_nmt = open(filename, 'w', encoding='utf-8')
    elif epoch == 0:
        f_out_nmt = open(nmt_result_filename, 'w', encoding = 'utf-8')
    else:
        filename = nmt_result_filename.split('.')[0] + '_' + str(epoch) + '.' + nmt_result_filename.split('.')[1]
        f_out_nmt = open(filename, 'w', encoding='utf-8')

    if seg_result_filename == None:
        filename = 'val_seg_result_' + str(epoch) + '.txt'    
        f_out_ctc = open(filename, 'w', encoding='utf-8')
    elif epoch == 0:
        f_out_ctc = open(seg_result_filename, 'w', encoding = 'utf-8')
    else:
        filename = seg_result_filename.split('.')[0] + '_' + str(epoch) + '.' + seg_result_filename.split('.')[1]
        f_out_ctc = open(filename, 'w', encoding='utf-8')

    model.eval()
    pad = EN_WORD.vocab.stoi['<pad>']
    total_loss_ctc = 0
    total_loss_nmt = 0
    ctc_loss = torch.nn.CTCLoss(blank=zh_word_size, reduction='mean')
    result_total_word = 0
    truth_total_word = 0
    total_correct = 0
    lenTotal = 0

    for b, batch in enumerate(val_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        ctc, len_ctc = batch.ctc
        with torch.no_grad():
            src = Variable(src.data.cuda())
            trg = Variable(trg.data.cuda())
            ctc = Variable(ctc.data.cuda())
        
        loss_ctc = 0
        srcTemp = torch.transpose(src, 0, 1)
        ctcTemp = torch.transpose(ctc, 0, 1)
        for i in range(len(srcTemp)):
            len_src_one = len_src[i]
            src_one = srcTemp[i][ : len_src_one]
            src_one = src_one.view(-1, 1)
            len_ctc_one = len_ctc[i]
            ctc_one = ctcTemp[i][ : len_ctc_one]
            ctc_one = ctc_one.view(-1, 1)
            len_src_one = len_src_one.view(-1, 1)
            len_ctc_one = len_ctc_one.view(-1, 1)
            output_ctc, hidden_ctc = model(src_one, ctc_one, teacher_forcing_ratio=0.0, is_twoLoss_ctc=True)
            log_probs = output_ctc.log_softmax(2)
            ctc_one = torch.transpose(ctc_one, 0, 1)
            loss_temp = ctc_loss(log_probs, ctc_one, len_src_one, len_ctc_one)
            #判断loss是否是nan或者是inf
            if loss_temp != loss_temp or loss_temp.data > 1000:
                #pdb.set_trace()
                print('---------Evaluate Loss is big.------------------')
                continue
            lenTotal = lenTotal + 1
            loss_ctc += loss_temp.data
            #ctc解码获取分词结果
            decoder = GreedyDecoder(ZH_WORD, space_idx=-1, blank_index=zh_word_size)

            decoded = decoder.decode(log_probs, len_src_one)
            targets = decoder._unflatten_targets(ctc_one, len_ctc_one)
            labels = decoder._process_strings(decoder._convert_to_strings(targets))

            f_out_ctc.write('Decoded: ' + ' '.join(decoded) + '\n')
            f_out_ctc.write('Targets: ' + ' '.join(labels) + '\n')
            f_out_ctc.flush()

            #pdb.set_trace()
            result_word, truth_word, correct = decoder.calWordNumber(decoded, labels)
            result_total_word = result_total_word + result_word
            truth_total_word = truth_total_word + truth_word
            total_correct = total_correct + correct

        total_loss_ctc += loss_ctc

        output, hidden = model(src, trg, teacher_forcing_ratio=0.0, is_twoLoss_ctc=False)
        loss_nmt = F.nll_loss(output[1:].view(-1, en_word_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=pad)

        total_loss_nmt += loss_nmt.data


        #nmt解码获取翻译结果
        decoder = NMTDecoder(EN_WORD, space_idx=-1, blank_index=en_word_size)

        decoded = decoder.decode(output, len_trg)
        trg = torch.transpose(trg, 0, 1)
        targets = decoder._unflatten_targets(trg, len_trg)
        labels = decoder._process_strings(decoder._convert_to_strings(targets))

        for i in range(len(decoded)):
            f_out_nmt.write('Decoded: ' + str(decoded[i]) + '\n')
            f_out_nmt.write('Targets: ' + str(labels[i]) + '\n')

        f_out_nmt.flush()

    f_out_nmt.close()
    f_out_ctc.close()

    precision = total_correct / result_total_word
    recall = total_correct / truth_total_word
    f_measure = 2 * precision * recall / (precision + recall)

    return total_loss_ctc / lenTotal, total_loss_nmt / len(val_iter), precision, recall, f_measure
    #return total_loss_ctc / len(val_iter), total_loss_nmt / len(val_iter), precision, recall, f_measure


def train_ctc(e, f_out, f_inf, model, optimizer, train_iter, grad_clip, zh_word_size, ZH_CHA, ZH_WORD):
    model.train()
    total_loss = 0
    ctc_loss = torch.nn.CTCLoss(blank=zh_word_size, reduction='mean')
    for b, batch in enumerate(train_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src, trg = src.cuda(), trg.cuda()
        optimizer.zero_grad()
        output, hidden = model(src, trg)
        #pdb.set_trace()
        log_probs = output.log_softmax(2)
        trg = torch.transpose(trg, 0, 1)
        loss = ctc_loss(log_probs, trg, len_src, len_trg)
        #判断loss是否是nan或者是inf
        if loss != loss or loss.data > 1000:
            #pdb.set_trace()
            string = "********ERROR: Loss is " + str(loss.data) + "\nInput src is " + str(src) + "\nInput target is " + str(trg) + '\n'
            f_inf.write(string)
            f_inf.flush()
            continue

        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.data

        if b % 500 == 0 and b != 0:
            total_loss = total_loss / 500
            print("[%d][loss:%5.2f]" %
                (b, total_loss))
            string = "[" + str(b) + "]\t[Loss]: " + str(total_loss) + '\n'
            f_out.write(string)
            f_out.flush()
            total_loss = 0

def train_nmt(e, f_out, model, optimizer, train_iter, vocab_size, grad_clip, ZH_WORD, EN_WORD):
    model.train()
    total_loss = 0
    pad = EN_WORD.vocab.stoi['<pad>']
    for b, batch in enumerate(train_iter):
        #pdb.set_trace()
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src, trg = src.cuda(), trg.cuda()
        optimizer.zero_grad()
        output = model(src, trg)
        loss = F.nll_loss(output[1:].view(-1, vocab_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=pad)
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.data

        if b % 1000 == 0 and b != 0:
            total_loss = total_loss / 1000
            print("[%d][loss:%5.2f][ppl:%5.2f]" %
                  (b, total_loss, math.exp(total_loss)))
            string = 'Batch: ' + str(b) + '\tLoss: ' + str(total_loss) + '\tPPL: ' + str(math.exp(total_loss)) + '\n'
            f_out.write(string)
            f_out.flush()
            total_loss = 0

def train_combine(e, f_out, model, optimizer, train_iter, vocab_size, grad_clip, ZH_CHA, EN_WORD):
    model.train()
    total_loss = 0
    pad = EN_WORD.vocab.stoi['<pad>']
    for b, batch in enumerate(train_iter):
        #pdb.set_trace()
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src, trg = src.cuda(), trg.cuda()
        optimizer.zero_grad()
        output, output_seg = model(src, trg)
        loss = F.nll_loss(output[1:].view(-1, vocab_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=pad)
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.data

        if b % 1000 == 0 and b != 0:
            total_loss = total_loss / 1000
            print("[%d][loss:%5.2f][ppl:%5.2f]" %
                  (b, total_loss, math.exp(total_loss)))
            string = 'Batch: ' + str(b) + '\tLoss: ' + str(total_loss) + '\tPPL: ' + str(math.exp(total_loss)) + '\n'
            f_out.write(string)
            f_out.flush()
            total_loss = 0

def train_update_twoLoss(e, f_out, f_inf, model, optimizer, train_iter, grad_clip, zh_word_size, en_word_size, ZH_CHA, ZH_WORD, EN_WORD):
    model.train()
    pad = EN_WORD.vocab.stoi['<pad>']
    total_loss_ctc = 0
    total_loss_nmt = 0
    ctc_loss = torch.nn.CTCLoss(blank=zh_word_size, reduction='mean')
    for b, batch in  enumerate(train_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        ctc, len_ctc = batch.ctc
        src, trg, ctc = src.cuda(), trg.cuda(), ctc.cuda()
        optimizer.zero_grad()

        loss_ctc = 0
        # pdb.set_trace()
        srcTemp = torch.transpose(src, 0, 1)
        ctcTemp = torch.transpose(ctc, 0, 1)
        for i in range(len(srcTemp)):
            len_src_one = len_src[i]
            src_one = srcTemp[i][ : len_src_one]
            src_one = src_one.view(-1, 1)
            len_src_one = len_src[i]
            len_ctc_one = len_ctc[i]
            ctc_one = ctcTemp[i][ : len_ctc_one]
            ctc_one = ctc_one.view(-1, 1)
            output_ctc, hidden_ctc = model(src_one, ctc_one, teacher_forcing_ratio=0.5, is_twoLoss_ctc=True)
            log_probs = output_ctc.log_softmax(2)
            ctc_one = torch.transpose(ctc_one, 0, 1)
            loss_temp = ctc_loss(log_probs, ctc_one, len_src_one, len_ctc_one)
            #判断loss是否是nan或者是inf
            if loss_temp != loss_temp or loss_temp.data > 1000:
                #pdb.set_trace()
                string = "********ERROR: Loss is " + str(loss_temp.data) + "\nInput src is " + str(src_one) + "\nInput target is " + str(ctc_one) + '\n'
                f_inf.write(string)
                f_inf.flush()
                continue
            loss_ctc += loss_temp
        # pdb.set_trace()
        
        output, hidden = model(src, trg, teacher_forcing_ratio=0.5, is_twoLoss_ctc=False)
        loss_nmt = F.nll_loss(output[1:].view(-1, en_word_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=pad)

        loss_ctc = loss_ctc / len(srcTemp)
        # loss_ctc.backward()
        # loss_nmt.backward()
        # loss = 5.0 * loss_ctc + loss_nmt
        loss = loss_ctc + 5.0 * loss_nmt
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss_ctc += loss_ctc.data
        total_loss_nmt += loss_nmt.data

        if b % 500 == 0 and b != 0:
            total_loss_ctc = total_loss_ctc / 500
            total_loss_nmt = total_loss_nmt / 500
            print("[%d][ctc_loss:%5.2f][[nmt_loss:%5.2f]]" %
                (b, total_loss_ctc, total_loss_nmt))
            string = "[" + str(b) + "]\t[CTC_Loss]: " + str(total_loss_ctc) + "\t[NMT_Loss]: " + str(total_loss_nmt) + '\n'
            f_out.write(string)
            f_out.flush()
            total_loss_ctc = 0
            total_loss_nmt = 0

def prepareDate(args, f_out):
    mode = args.mode
    print("The training mode is " + str(mode) + '.')
    print("Preparing dataset...")
    string = "The training mode is " + mode + '.\n'
    f_out.write(string)
    f_out.write("Preparing dataset...\n")
    f_out.flush()
    if mode == 'update_twoLoss':
        train_iter, val_iter, test_iter, INPUTField, OUTPUTField, CTCField = load_dataset(args)
        input_dict_size, output_dict_size, ctc_dict_size = len(INPUTField.vocab), len(OUTPUTField.vocab), len(CTCField.vocab)
        print("[TRAIN]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)"
            % (len(train_iter), len(train_iter.dataset),
            len(test_iter), len(test_iter.dataset)))
        print("[INPUT_vocab]:%d [OUTPUT_vocab]:%d [CTC_vocab]:%d" % (input_dict_size, output_dict_size, ctc_dict_size))
        string = "[TRAIN]:" + str(len(train_iter)) + "\t[TEST]:" + str(len(test_iter)) + '\n' \
            + "[INPUT_vocab]:" + str(input_dict_size) + '\t[OUTPUT_vocab]:' + str(output_dict_size) + '\t[CTC_vocab]:' + str(ctc_dict_size) + '\n' 
        f_out.write(string)
        f_out.flush()
        return train_iter, val_iter, test_iter, INPUTField, OUTPUTField, CTCField, input_dict_size, output_dict_size, ctc_dict_size
    else:
        train_iter, val_iter, test_iter, INPUTField, OUTPUTField = load_dataset(args)
        input_dict_size, output_dict_size = len(INPUTField.vocab), len(OUTPUTField.vocab)
        print("[TRAIN]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)"
            % (len(train_iter), len(train_iter.dataset),
            len(test_iter), len(test_iter.dataset)))
        print("[INPUT_vocab]:%d [OUTPUT_vocab]:%d" % (input_dict_size, output_dict_size))
        string = "[TRAIN]:" + str(len(train_iter)) + "\t[TEST]:" + str(len(test_iter)) + '\n' \
            + "[INPUT_vocab]:" + str(input_dict_size) + '\t[OUTPUT_vocab]:' + str(output_dict_size) + '\n' 
        f_out.write(string)
        f_out.flush()
        return train_iter, val_iter, test_iter, INPUTField, OUTPUTField, input_dict_size, output_dict_size

def main_ctc(args, f_out, f_inf):
    #准备数据
    train_iter, val_iter, test_iter, ZH_CHA, ZH_WORD, zh_cha_size, zh_word_size = prepareDate(args, f_out)

    #定义并初始化网络
    print("Instantiating models...")
    f_out.write("Instantiating models...\n")
    f_out.flush()

    encoder = Encoder_Combine(args.hidden_size, args.hidden_size,
        n_layers=args.nmt_encoder_layers, dropout=args.dropout, model_cell=args.model_cell)
    en_word_size = zh_word_size
    decoder = Decoder(args.embed_size, args.hidden_size, en_word_size,
        n_layers=args.nmt_decoder_layers, dropout=args.dropout, model_cell=args.model_cell)
    ctc_seg = CTC_Seg(zh_cha_size, args.embed_size, args.hidden_size, zh_word_size + 1, 
        n_layers=args.ctc_layers, dropout=args.dropout, model_cell=args.model_cell)
    seq2seq = Seq2Seq(args.mode, encoder, decoder, ctc_seg).cuda()
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq)

    #加载已经训练好的模型
    if args.model_path != None:
        print('Load the trained model ' + str(args.model_path) + ".")
        string = "Load the trained model " + str(args.model_path) + '\n'
        f_out.write(string)
        f_out.flush()
        seq2seq.load_state_dict(torch.load(args.model_path))

    #Test trained model
    if args.istest:
        if args.model_path == None:
            print('Please input the trained model to test.')
            exit(1)
        else:
            print('Test the model ' + str(args.model_path) + '.')
            test_loss, test_precision, test_recall, test_f_measure = evaluate_ctc(0, seq2seq, test_iter, zh_word_size, ZH_WORD, args.seg_result_file)
            print("[TEST] loss:%5.2f \t Precision:%5.2f \t Recall:%5.2f \t F_measure:%5.2f" % (test_loss, test_precision, test_recall, test_f_measure))
            string = 'Test loss: ' + str(test_loss) + '\tTest Precision: ' + str(test_precision) + '\tTest Recall: ' + str(test_recall) + '\tTest F_measure: ' + str(test_f_measure) +'\n'
            f_out.write(string)
            f_out.flush()
            exit(0)
        
    #记录开发集上loss上升次数的变量
    increasingNumber = 0
    best_model_path = ""
    best_val_loss = None

    for e in range(1, args.epochs+1):
        train_ctc(e, f_out, f_inf, seq2seq, optimizer, train_iter, args.grad_clip, zh_word_size, ZH_CHA, ZH_WORD)
        val_loss, val_precision, val_recall, val_f_measure = evaluate_ctc(e, seq2seq, val_iter, zh_word_size, ZH_WORD, args.val_seg_result_file)
        print("[Epoch:%d] val_loss:%5.3f | val_ppl:%5.2f | val_precision:%5.2f | val_recall:%5.2f | val_f_measure:%5.2f"
            % (e, val_loss, math.exp(val_loss), val_precision, val_recall, val_f_measure))
        f_out.write('---------------------------------\n')
        string = 'Epoch: ' + str(e) + '\tval_loss: ' + str(val_loss) + '\tval_PPL: ' + str(math.exp(val_loss)) + '\tval_precision: ' + str(val_precision) + '\tval_recall: ' + str(val_recall) + '\tval_f_measure: ' + str(val_f_measure) + '\n'
        f_out.write(string)
        f_out.write('---------------------------------\n')
        f_out.flush()

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("Saving model...")
            f_out.write("Saving model...\n")
            f_out.flush()
            if not os.path.isdir(args.save_path):
                os.makedirs(args.save_path)
            torch.save(seq2seq.state_dict(), './%s/seq2seq_%d.pt' % (args.save_path, e))
            best_val_loss = val_loss
            best_model_path = './' + args.save_path + '/seq2seq_' + str(e) + '.pt'
            best_val_loss = val_loss
        elif val_loss > best_val_loss:
            increasingNumber = increasingNumber + 1
            if increasingNumber == 5:
                print('The loss of valid set has risen ' + str(increasingNumber) + ' times.')
                string = 'The loss of valid set has risen ' + str(increasingNumber) + ' times.\n'
                f_out.write(string)
                f_out.flush()
                break
    if best_model_path == "":
        print(str(best_model_path) + " isn't existing.")
        return

    #加载在开发及上性能最好的模型进行测试
    seq2seq.load_state_dict(torch.load(best_model_path))
    test_loss, test_precision, test_recall, test_f_measure = evaluate_ctc(0, seq2seq, test_iter, zh_word_size, ZH_WORD, args.seg_result_file)
    print("[TEST] loss:%5.2f \t Precision:%5.2f \t Recall:%5.2f \t F_measure:%5.2f" % (test_loss, test_precision, test_recall, test_f_measure))
    string = 'Test loss: ' + str(test_loss) + '\tTest Precision: ' + str(test_precision) + '\tTest Recall: ' + str(test_recall) + '\tTest F_measure: ' + str(test_f_measure) +'\n'
    f_out.write(string)
    f_out.flush()


def main_nmt(args, f_out):
    #准备数据
    train_iter, val_iter, test_iter, ZH, EN_WORD, zh_size, en_word_size = prepareDate(args, f_out)

    print("Instantiating models...")
    f_out.write("Instantiating models...\n")
    f_out.flush()
    encoder = Encoder(zh_size, args.embed_size, args.hidden_size,
                      n_layers=args.nmt_encoder_layers, dropout=args.dropout, model_cell=args.model_cell)
    decoder = Decoder(args.embed_size, args.hidden_size, en_word_size,
                      n_layers=args.nmt_decoder_layers, dropout=args.dropout, model_cell=args.model_cell)
    ctc_seg = None
    seq2seq = Seq2Seq(args.mode, encoder, decoder, ctc_seg).cuda()
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq)

    #加载已经训练好的模型
    if args.model_path != None:
        print('Load the trained model ' + str(args.model_path) + ".")
        string = "Load the trained model " + str(args.model_path) + '\n'
        f_out.write(string)
        f_out.flush()
        seq2seq.load_state_dict(torch.load(args.model_path))
  
    #Test trained model
    if args.istest:
        if args.model_path == None:
            print('Please input the trained model to test.')
            exit(1)
        else:
            print('Test the model ' + str(args.model_path) + '.')
            test_loss = evaluate_nmt(0, seq2seq, test_iter, en_word_size, ZH, EN_WORD, args.nmt_result_file)
            print("[TEST] loss:%5.2f" % test_loss)
            string = 'Test loss: ' + str(test_loss) + '\n'
            f_out.write(string)
            f_out.flush()
            exit(0)
            
    #记录开发集上loss上升次数的变量
    increasingNumber = 0
    best_model_path = ""
    best_val_loss = None

    for e in range(1, args.epochs+1):
        train_nmt(e, f_out, seq2seq, optimizer, train_iter,
            en_word_size, args.grad_clip, ZH, EN_WORD)
        val_loss = evaluate_nmt(e, seq2seq, val_iter, en_word_size, ZH, EN_WORD, args.val_nmt_result_file)

        print("[Epoch:%d] val_loss:%5.3f | val_ppl:%5.2f"
            % (e, val_loss, math.exp(val_loss)))
        f_out.write('---------------------------------\n')
        string = 'Epoch: ' + str(e) + '\tval_loss: ' + str(val_loss) + '\tval_PPL: ' + str(math.exp(val_loss)) + '\n'
        f_out.write(string)
        f_out.write('---------------------------------\n')
        f_out.flush()

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("Saving model...")
            f_out.write("Saving model...\n")
            f_out.flush()
            if not os.path.isdir(args.save_path):
                os.makedirs(args.save_path)
            torch.save(seq2seq.state_dict(), './%s/seq2seq_%d.pt' % (args.save_path, e))
            best_model_path = './' + args.save_path + '/seq2seq_' + str(e) + '.pt'
            best_val_loss = val_loss
        elif val_loss > best_val_loss:
            increasingNumber = increasingNumber + 1
            if best_model_path != "":
                seq2seq.load_state_dict(torch.load(best_model_path))
            if increasingNumber == 5:
                print('The loss of valid set has risen ' + str(increasingNumber) + ' times.')
                string = 'The loss of valid set has risen ' + str(increasingNumber) + ' times.\n'
                f_out.write(string)
                f_out.flush()
                break
    if best_model_path == "":
        print(str(best_model_path) + " isn't existing.")
        return

    seq2seq.load_state_dict(torch.load(best_model_path))
    test_loss = evaluate_nmt(0, seq2seq, test_iter, en_word_size, ZH, EN_WORD, args.nmt_result_file)
    print("[TEST] loss:%5.2f" % test_loss)
    string = 'Test loss: ' + str(test_loss) + '\n'
    f_out.write(string)
    f_out.flush()

def main_combine(args, f_out):
    #准备数据
    train_iter, val_iter, test_iter, ZH_CHA, EN_WORD, zh_cha_size, en_word_size = prepareDate(args, f_out)

    #pdb.set_trace()
    print("Instantiating models...")
    f_out.write("Instantiating models...\n")
    f_out.flush()
    encoder = Encoder_Combine(args.hidden_size, args.hidden_size,
        n_layers=args.nmt_encoder_layers, dropout=args.dropout, model_cell=args.model_cell)
    decoder = Decoder(args.embed_size, args.hidden_size, en_word_size,
        n_layers=args.nmt_decoder_layers, dropout=args.dropout, model_cell=args.model_cell)
    zh_word_size = en_word_size
    ctc_seg = CTC_Seg(zh_cha_size, args.embed_size, args.hidden_size, zh_word_size + 1, 
        n_layers=args.ctc_layers, dropout=args.dropout, model_cell=args.model_cell)
    
    seq2seq = Seq2Seq(args.mode, encoder, decoder, ctc_seg).cuda()
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq)

    #如果存在初始的CTC_Seg model，则将其加载作为整个网络中CTC分词部分的初始化
    if args.init_CTC_model != None:
        print('Load the initial CTC_Seg model ' + str(args.init_CTC_model) + ".")
        string = "Load the initial CTC_Seg model " + str(args.init_CTC_model) + '\n'
        f_out.write(string)
        f_out.flush()
        seq2seq.load_state_dict(torch.load(args.init_CTC_model))

    #加载已经训练好的模型
    if args.model_path != None:
        print('Load the trained model ' + str(args.model_path) + ".")
        string = "Load the trained model " + str(args.model_path) + '\n'
        f_out.write(string)
        f_out.flush()
        seq2seq.load_state_dict(torch.load(args.model_path))
    
    #Test trained model
    if args.istest:
        if args.model_path == None:
            print('Please input the trained model to test.')
            exit(1)
        else:
            print('Test the model ' + str(args.model_path) + '.')
            test_loss = evaluate_combine(0, seq2seq, test_iter, en_word_size, ZH_CHA, EN_WORD, args.nmt_result_file)
            print("[TEST] loss:%5.2f" % test_loss)
            string = 'Test loss: ' + str(test_loss) + '\n'
            f_out.write(string)
            f_out.flush()
            exit(0)

    #记录开发集上loss上升次数的变量
    increasingNumber = 0
    best_model_path = ""
    best_val_loss = None

    for e in range(1, args.epochs+1):
        train_combine(e, f_out, seq2seq, optimizer, train_iter,
            en_word_size, args.grad_clip, ZH_CHA, EN_WORD)
        val_loss = evaluate_combine(e, seq2seq, val_iter, en_word_size, ZH_CHA, EN_WORD, args.val_nmt_result_file)

        print("[Epoch:%d] val_loss:%5.3f | val_ppl:%5.2f"
            % (e, val_loss, math.exp(val_loss)))
        f_out.write('---------------------------------\n')
        string = 'Epoch: ' + str(e) + '\tval_loss: ' + str(val_loss) + '\tval_PPL: ' + str(math.exp(val_loss)) + '\n'
        f_out.write(string)
        f_out.write('---------------------------------\n')
        f_out.flush()

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("Saving model...")
            f_out.write("Saving model...\n")
            f_out.flush()
            if not os.path.isdir(args.save_path):
                os.makedirs(args.save_path)
            torch.save(seq2seq.state_dict(), './%s/seq2seq_%d.pt' % (args.save_path, e))
            best_model_path = './' + args.save_path + '/seq2seq_' + str(e) + '.pt'
            best_val_loss = val_loss
        elif val_loss > best_val_loss:
            increasingNumber = increasingNumber + 1
            if best_model_path != "":
                seq2seq.load_state_dict(torch.load(best_model_path))
            if increasingNumber == 5:
                print('The loss of valid set has risen ' + str(increasingNumber) + ' times.')
                string = 'The loss of valid set has risen ' + str(increasingNumber) + ' times.\n'
                f_out.write(string)
                f_out.flush()
                break
    if best_model_path == "":
        print(str(best_model_path) + " isn't existing.")
        return

    seq2seq.load_state_dict(torch.load(best_model_path))
    test_loss = evaluate_combine(0, seq2seq, test_iter, en_word_size, ZH_CHA, EN_WORD, args.nmt_result_file)
    print("[TEST] loss:%5.2f" % test_loss)
    string = 'Test loss: ' + str(test_loss) + '\n'
    f_out.write(string)
    f_out.flush()

def main_refine(args, f_out, f_inf):
    #准备数据
    train_iter, val_iter, test_iter, ZH_CHA, ZH_WORD, zh_cha_size, zh_word_size = prepareDate(args, f_out)

    #定义并初始化网络
    print("Instantiating models...")
    f_out.write("Instantiating models...\n")
    f_out.flush()

    encoder = Encoder_Combine(args.hidden_size, args.hidden_size,
        n_layers=args.nmt_encoder_layers, dropout=args.dropout, model_cell=args.model_cell)
    en_word_size = zh_word_size
    decoder = Decoder(args.embed_size, args.hidden_size, en_word_size,
        n_layers=args.nmt_decoder_layers, dropout=args.dropout, model_cell=args.model_cell)
    ctc_seg = CTC_Seg(zh_cha_size, args.embed_size, args.hidden_size, zh_word_size + 1, 
        n_layers=args.ctc_layers, dropout=args.dropout, model_cell=args.model_cell)
    seq2seq = Seq2Seq(args.mode, encoder, decoder, ctc_seg).cuda()
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq)

    #加载combine之后的模型
    if args.init_combine_model != None:
        print('Load the initial combine model ' + str(args.init_combine_model) + ".")
        string = "Load the trained model " + str(args.init_combine_model) + '\n'
        f_out.write(string)
        f_out.flush()
        seq2seq.load_state_dict(torch.load(args.init_combine_model))

    #加载已经训练好的模型
    if args.model_path != None:
        print('Load the trained model ' + str(args.model_path) + ".")
        string = "Load the trained model " + str(args.model_path) + '\n'
        f_out.write(string)
        f_out.flush()
        seq2seq.load_state_dict(torch.load(args.model_path))

    #Test trained model
    if args.istest:
        if args.model_path == None:
            print('Please input the trained model to test.')
            exit(1)
        else:
            print('Test the model ' + str(args.model_path) + '.')
            test_loss, test_precision, test_recall, test_f_measure = evaluate_ctc(0, seq2seq, test_iter, zh_word_size, ZH_WORD, args.seg_result_file)
            print("[TEST] loss:%5.2f \t Precision:%5.2f \t Recall:%5.2f \t F_measure:%5.2f" % (test_loss, test_precision, test_recall, test_f_measure))
            string = 'Test loss: ' + str(test_loss) + '\tTest Precision: ' + str(test_precision) + '\tTest Recall: ' + str(test_recall) + '\tTest F_measure: ' + str(test_f_measure) +'\n'
            f_out.write(string)
            f_out.flush()
            exit(0)

    #记录开发集上loss上升次数的变量
    increasingNumber = 0
    best_model_path = ""
    best_val_loss = None

    for e in range(1, args.epochs+1):
        train_ctc(e, f_out, f_inf, seq2seq, optimizer, train_iter, args.grad_clip, zh_word_size, ZH_CHA, ZH_WORD)
        val_loss, val_precision, val_recall, val_f_measure = evaluate_ctc(e, seq2seq, val_iter, zh_word_size, ZH_WORD, args.val_seg_result_file)
        print("[Epoch:%d] val_loss:%5.3f | val_ppl:%5.2f | val_precision:%5.2f | val_recall:%5.2f | val_f_measure:%5.2f"
            % (e, val_loss, math.exp(val_loss), val_precision, val_recall, val_f_measure))
        f_out.write('---------------------------------\n')
        string = 'Epoch: ' + str(e) + '\tval_loss: ' + str(val_loss) + '\tval_PPL: ' + str(math.exp(val_loss)) + '\tval_precision: ' + str(val_precision) + '\tval_recall: ' + str(val_recall) + '\tval_f_measure: ' + str(val_f_measure) + '\n'
        f_out.write(string)
        f_out.write('---------------------------------\n')
        f_out.flush()

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("Saving model...")
            f_out.write("Saving model...\n")
            f_out.flush()
            if not os.path.isdir(args.save_path):
                os.makedirs(args.save_path)
            torch.save(seq2seq.state_dict(), './%s/seq2seq_%d.pt' % (args.save_path, e))
            best_val_loss = val_loss
            best_model_path = './' + args.save_path + '/seq2seq_' + str(e) + '.pt'
            best_val_loss = val_loss
        elif val_loss > best_val_loss:
            increasingNumber = increasingNumber + 1
            if increasingNumber == 5:
                print('The loss of valid set has risen ' + str(increasingNumber) + ' times.')
                string = 'The loss of valid set has risen ' + str(increasingNumber) + ' times.\n'
                f_out.write(string)
                f_out.flush()
                break
    if best_model_path == "":
        print(str(best_model_path) + " isn't existing.")
        return

    #加载在开发及上性能最好的模型进行测试
    seq2seq.load_state_dict(torch.load(best_model_path))
    test_loss, test_precision, test_recall, test_f_measure = evaluate_ctc(0, seq2seq, test_iter, zh_word_size, ZH_WORD, args.seg_result_file)
    print("[TEST] loss:%5.2f \t Precision:%5.2f \t Recall:%5.2f \t F_measure:%5.2f" % (test_loss, test_precision, test_recall, test_f_measure))
    string = 'Test loss: ' + str(test_loss) + '\tTest Precision: ' + str(test_precision) + '\tTest Recall: ' + str(test_recall) + '\tTest F_measure: ' + str(test_f_measure) +'\n'
    f_out.write(string)
    f_out.flush()

def main_update_twoLoss(args, f_out, f_inf):
    #准备数据，加载训练ctc分词网络的数据（中文字-中文词），和联合训练的数据（中文字-英文词）
    train_iter, val_iter, test_iter, ZH_CHA, EN_WORD, ZH_WORD, zh_cha_size, en_word_size, zh_word_size = prepareDate(args, f_out)

   #定义并初始化网络
    print("Instantiating models...")
    f_out.write("Instantiating models...\n")
    f_out.flush()

    encoder = Encoder_Combine(args.hidden_size, args.hidden_size,
        n_layers=args.nmt_encoder_layers, dropout=args.dropout, model_cell=args.model_cell)
    decoder = Decoder(args.embed_size, args.hidden_size, en_word_size,
        n_layers=args.nmt_decoder_layers, dropout=args.dropout, model_cell=args.model_cell)
    ctc_seg = CTC_Seg(zh_cha_size, args.embed_size, args.hidden_size, zh_word_size + 1, 
        n_layers=args.ctc_layers, dropout=args.dropout, model_cell=args.model_cell)
    seq2seq = Seq2Seq(args.mode, encoder, decoder, ctc_seg).cuda()
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq)

    #如果存在初始的CTC_Seg model，则将其加载作为整个网络中CTC分词部分的初始化
    if args.init_CTC_model != None:
        print('Load the initial CTC_Seg model ' + str(args.init_CTC_model) + ".")
        string = "Load the initial CTC_Seg model " + str(args.init_CTC_model) + '\n'
        f_out.write(string)
        f_out.flush()
        seq2seq.load_state_dict(torch.load(args.init_CTC_model))

    #加载combine之后的模型
    if args.init_combine_model != None:
        print('Load the initial combine model ' + str(args.init_combine_model) + ".")
        string = "Load the trained model " + str(args.init_combine_model) + '\n'
        f_out.write(string)
        f_out.flush()
        seq2seq.load_state_dict(torch.load(args.init_combine_model))

    #加载已经训练好的模型
    if args.model_path != None:
        print('Load the trained model ' + str(args.model_path) + ".")
        string = "Load the trained model " + str(args.model_path) + '\n'
        f_out.write(string)
        f_out.flush()
        seq2seq.load_state_dict(torch.load(args.model_path))

    #Test trained model
    if args.istest:
        if args.model_path == None:
            print('Please input the trained model to test.')
            exit(1)
        else:
            print('Test the model ' + str(args.model_path) + '.')
            test_loss_ctc, test_loss_nmt, test_precision, test_recall, test_f_measure = evaluate_update_twoLoss(0, seq2seq, test_iter, zh_word_size, en_word_size, ZH_CHA, ZH_WORD, EN_WORD, args.seg_result_file, args.nmt_result_file)
            print("[TEST] nmt_loss:%5.3f | ctc_loss:%5.3f | test_precision:%5.3f | test_recall:%5.3f | test_f_measure:%5.3f " % (test_loss_nmt, test_loss_ctc, test_precision, test_recall, test_f_measure))
            string = 'Test nmt_loss: ' + str(test_loss_nmt) + '\tTest ctc_loss: ' + str(test_loss_ctc) + '\tTest precision: ' + str(test_precision) + '\tTest recall: ' + str(test_recall) + '\tTest f_measure: ' + str(test_f_measure) + '\n'
            f_out.write(string)
            f_out.flush()
            exit(0)

    #记录开发集上loss上升次数的变量
    increasingNumber = 0
    best_model_path = ""
    best_val_nmt_loss = None
    best_val_ctc_loss = None

    for e in range(1, args.epochs+1):
        train_update_twoLoss(e, f_out, f_inf, seq2seq, optimizer, train_iter, args.grad_clip, zh_word_size, en_word_size, ZH_CHA, ZH_WORD, EN_WORD)
        val_loss_ctc, val_loss_nmt, val_precision, val_recall, val_f_measure = evaluate_update_twoLoss(e, seq2seq, val_iter, zh_word_size, en_word_size, ZH_CHA, ZH_WORD, EN_WORD, args.val_seg_result_file, args.val_nmt_result_file)
        print("[Epoch:%d] | val_loss_nmt:%5.3f | val_loss_ctc:%5.3f | val_precision:%5.3f | val_recall:%5.3f | val_f_measure:%5.3f "
            % (e, val_loss_nmt, val_loss_ctc, val_precision, val_recall, val_f_measure))
        f_out.write('---------------------------------\n')
        string = 'Epoch: ' + str(e) + '\tval_loss_nmt: ' + str(val_loss_nmt) + '\tval_loss_ctc: ' + str(val_loss_ctc) + '\tval_precision: ' + str(val_precision) + '\tval_recall: ' + str(val_recall) + '\tval_f_measure: ' + str(val_f_measure) +  '\n'
        f_out.write(string)
        f_out.write('---------------------------------\n')
        f_out.flush()

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_nmt_loss or val_loss_nmt < best_val_nmt_loss or not best_val_ctc_loss or val_loss_ctc < best_val_ctc_loss:
            print("Saving model...")
            f_out.write("Saving model...\n")
            f_out.flush()
            if not os.path.isdir(args.save_path):
                os.makedirs(args.save_path)
            torch.save(seq2seq.state_dict(), './%s/seq2seq_%d.pt' % (args.save_path, e))
            best_val_nmt_loss = val_loss_nmt
            best_val_ctc_loss = val_loss_ctc
            best_model_path = './' + args.save_path + '/seq2seq_' + str(e) + '.pt'
        elif val_loss_nmt > best_val_nmt_loss and val_loss_ctc > best_val_ctc_loss:
            increasingNumber = increasingNumber + 1
            if increasingNumber == 5:
                print('The loss of valid set has risen ' + str(increasingNumber) + ' times.')
                string = 'The loss of valid set has risen ' + str(increasingNumber) + ' times.\n'
                f_out.write(string)
                f_out.flush()
                break
    if best_model_path == "":
        print(str(best_model_path) + " isn't existing.")
        return

    #加载在开发及上性能最好的模型进行测试
    seq2seq.load_state_dict(torch.load(best_model_path))
    test_loss_ctc, test_loss_nmt, test_precision, test_recall, test_f_measure = evaluate_update_twoLoss(0, seq2seq, test_iter, zh_word_size, en_word_size, ZH_CHA, ZH_WORD, EN_WORD, args.seg_result_file, args.nmt_result_file)
    print("[TEST] nmt_loss:%5.3f | ctc_loss:%5.3f | test_precision:%5.3f | test_recall:%5.3f | test_f_measure:%5.3f " % (test_loss_nmt, test_loss_ctc, test_precision, test_recall, test_f_measure))
    string = 'Test nmt_loss: ' + str(test_loss_nmt) + '\tTest ctc_loss: ' + str(test_loss_ctc) + '\tTest precision: ' + str(test_precision) + '\tTest recall: ' + str(test_recall) + '\tTest f_measure: ' + str(test_f_measure) + '\n'
    f_out.write(string)
    f_out.flush()


def main():
    args = parse_arguments()
    f_out = open(args.log_FileName, 'w', encoding='utf-8-sig')
    assert torch.cuda.is_available()

    if args.mode == 'ctc':
        with open('infSentence', 'w', encoding = 'utf-8') as f_inf:
            main_ctc(args, f_out, f_inf)
    elif args.mode == 'nmt':
        main_nmt(args, f_out)
    elif args.mode == 'nmt_char':
        main_nmt(args, f_out)
    elif args.mode == 'combine':
        main_combine(args, f_out)
    elif args.mode == 'refine_ctc':
        with open('infSentence', 'w', encoding = 'utf-8') as f_inf:
            main_refine(args, f_out, f_inf)
    elif args.mode == 'update_twoLoss':
        with open('infSentence', 'w', encoding = 'utf-8') as f_inf:
            main_update_twoLoss(args, f_out, f_inf)
    else:
        print('Please input correct training mode. Such as: ctc | nmt | nmt_char | combine | refine_ctc | update_twoLoss')
    f_out.close()

if __name__ == "__main__":
    #argvLen = len(sys.argv)
    #if argvLen <= 1:
    #    print('<Usage>: python train.py logFilenName. \n')
    #    exit(1)
    #logFilenName = argv[1]
    try:
        main()
        #with open(logFilenName, 'w', encoding='utf-8-sig') as f_out:
        #    main(f_out)
    except KeyboardInterrupt as e:
        print("[STOP]", e)
