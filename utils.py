#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
    加载数据，处理数据等的基本操作文件
    Author: Lihui Wang     
    Date: 2019-02-25
''' 

import re
from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset as Trans
import nltk.tokenize.punkt
import nltk
from myDataset import MyDataSet as mydataset
import pdb

'''
    load data
'''
def load_dataset(args):
    #tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    
    #def tokenize_enword(text):
    #    return [tok for tok in text.strip().split()]

    def tokenzie_cncha(text):
        #return [tok for tok in re.sub('\s','',text).strip()]
        return [tok for tok in text.strip()]

    def tokenzie_cnword(text):
        return [tok for tok in text.strip().split()]

    def tokenzie_enword(text):
        return tokenizer.tokenize(text)


    ZH_CHA = Field(tokenize=tokenzie_cncha, include_lengths=True,
            init_token='<sos>', eos_token='<eos>')

    ZH_WORD = Field(tokenzie_cnword, include_lengths=True,
            init_token='<sos>', eos_token='<eos>')

    EN_WORD = Field(tokenize=tokenzie_enword, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')

    #pdb.set_trace()

    #According to training mode, load data
    if args.mode == 'ctc':
        exts=(args.extension.split()[0], args.extension.split()[1])
        train, val, test = Trans.splits(path=args.path, exts=exts,
             fields=(ZH_CHA, ZH_WORD), train=args.train, validation=args.valid, test=args.test)

        ZH_CHA.build_vocab(train.src)
        ZH_WORD.build_vocab(train.trg, max_size=50000)
        
        train_iter, val_iter, test_iter = BucketIterator.splits(
                (train, val, test), batch_size=args.batch_size, repeat=False)
        return train_iter, val_iter, test_iter, ZH_CHA, ZH_WORD

    elif args.mode == 'nmt':
        exts=(args.extension.split()[0], args.extension.split()[1])
        train, val, test = Trans.splits(path=args.path, exts=exts,
             fields=(ZH_WORD, EN_WORD), train=args.train, validation=args.valid, test=args.test)

        ZH_WORD.build_vocab(train.src, max_size=50000)
        EN_WORD.build_vocab(train.trg, max_size=50000)

        train_iter, val_iter, test_iter = BucketIterator.splits(
                (train, val, test), batch_size=args.batch_size, repeat=False)
        return train_iter, val_iter, test_iter, ZH_WORD, EN_WORD

    elif args.mode == 'nmt_char':
        exts=(args.extension.split()[0], args.extension.split()[1])
        train, val, test = Trans.splits(path=args.path, exts=exts,
             fields=(ZH_CHA, EN_WORD), train=args.train, validation=args.valid, test=args.test)

        ZH_CHA.build_vocab(train.src)
        EN_WORD.build_vocab(train.trg, max_size=50000)

        train_iter, val_iter, test_iter = BucketIterator.splits(
                (train, val, test), batch_size=args.batch_size, repeat=False)
        return train_iter, val_iter, test_iter, ZH_CHA, EN_WORD
    
    elif args.mode == 'combine':
        exts=(args.extension.split()[0], args.extension.split()[1])
        train, val, test = Trans.splits(path=args.path, exts=exts,
             fields=(ZH_CHA, EN_WORD), train=args.train, validation=args.valid, test=args.test)

        ZH_CHA.build_vocab(train.src)
        EN_WORD.build_vocab(train.trg, max_size=50000)


        train_iter, val_iter, test_iter = BucketIterator.splits(
                (train, val, test), batch_size=args.batch_size, repeat=False)
        return train_iter, val_iter, test_iter, ZH_CHA, EN_WORD

    elif args.mode == 'refine_ctc':
        exts=(args.extension.split()[0], args.extension.split()[1])
        train, val, test = Trans.splits(path=args.path, exts=exts,
             fields=(ZH_CHA, ZH_WORD), train=args.train, validation=args.valid, test=args.test)

        ZH_CHA.build_vocab(train.src)
        ZH_WORD.build_vocab(train.trg, max_size=50000)
        
        train_iter, val_iter, test_iter = BucketIterator.splits(
                (train, val, test), batch_size=args.batch_size, repeat=False)
        return train_iter, val_iter, test_iter, ZH_CHA, ZH_WORD

    elif args.mode == 'update_twoLoss':
        exts = (args.extension.split()[0], args.extension.split()[1], args.extension.split()[2])
        train, val, test, = mydataset.splits(path=args.path, exts=exts, fields=(ZH_CHA, EN_WORD, ZH_WORD), train=args.train, validation=args.valid, test=args.test)
        ZH_CHA.build_vocab(train.src)
        EN_WORD.build_vocab(train.trg, max_size=50000)
        ZH_WORD.build_vocab(train.ctc, max_size=50000)

        train_iter, val_iter, test_iter = BucketIterator.splits(
                (train, val, test), batch_size=args.batch_size, repeat=False)

        return train_iter, val_iter, test_iter, ZH_CHA, EN_WORD, ZH_WORD