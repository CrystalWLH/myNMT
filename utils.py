#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
    Load data, process data, tokenize, segementation and so on.
    Author: Lihui Wang     
    Create Date: 2019-02-25
    Update Date: 2019-12-18
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

    def tokenzie_ticha(text):
        return [tok for tok in text.strip().split()]
    
    def tokenzie_tiword(text):
        return [tok for tok in text.strip().split()]
    
    ZH_CHA = Field(tokenize=tokenzie_cncha, include_lengths=True,
            init_token='<sos>', eos_token='<eos>')

    ZH_WORD = Field(tokenzie_cnword, include_lengths=True,
            init_token='<sos>', eos_token='<eos>')

    EN_WORD = Field(tokenize=tokenzie_enword, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    
    TI_CHA = Field(tokenize=tokenzie_ticha, include_lengths=True,
            init_token='<sos>', eos_token='<eos>')

    TI_WORD = Field(tokenize=tokenzie_tiword, include_lengths=True,
            init_token='<sos>', eos_token='<eos>')

    #pdb.set_trace()

    #According to training mode, load data
    if args.mode == 'ctc':
        if args.extension.split()[0] == '.ti':
            exts=(args.extension.split()[0], args.extension.split()[1])
            train, val, test = Trans.splits(path=args.path, exts=exts,
                fields=(TI_CHA, TI_WORD), train=args.train, validation=args.valid, test=args.test)
            if args.src_dict_maxSize == -1:
                TI_CHA.build_vocab(train.src)
            else:
                TI_CHA.build_vocab(train.src, max_size=args.src_dict_maxSize)
            if args.seg_dict_maxSize == -1:
                TI_WORD.build_vocab(train.trg)
            else:
                TI_WORD.build_vocab(train.trg, max_size=args.seg_dict_maxSize)
            train_iter, val_iter, test_iter = BucketIterator.splits(
                    (train, val, test), batch_size=args.batch_size, repeat=False)
            return train_iter, val_iter, test_iter, TI_CHA, TI_WORD
        else:
            exts=(args.extension.split()[0], args.extension.split()[1])
            train, val, test = Trans.splits(path=args.path, exts=exts,
                fields=(ZH_CHA, ZH_WORD), train=args.train, validation=args.valid, test=args.test)

            if args.src_dict_maxSize == -1:
                ZH_CHA.build_vocab(train.src)
            else:
                ZH_CHA.build_vocab(train.src, max_size=args.src_dict_maxSize)
            if args.seg_dict_maxSize == -1:
                ZH_WORD.build_vocab(train.trg)
            else:
                ZH_WORD.build_vocab(train.trg, max_size=args.seg_dict_maxSize)
        
            train_iter, val_iter, test_iter = BucketIterator.splits(
                    (train, val, test), batch_size=args.batch_size, repeat=False)
            return train_iter, val_iter, test_iter, ZH_CHA, ZH_WORD

    elif args.mode == 'nmt':
        if args.extension.split()[0] == '.ti' or args.extension.split()[1] == '.zh':
            exts=(args.extension.split()[0], args.extension.split()[1])
            train, val, test = Trans.splits(path=args.path, exts=exts,
                fields=(TI_WORD, ZH_WORD), train=args.train, validation=args.valid, test=args.test)
            if args.seg_dict_maxSize == -1:
                TI_WORD.build_vocab(train.src)
            else:
                TI_WORD.build_vocab(train.src, max_size=args.seg_dict_maxSize)
            if args.tgt_dict_maxSize == -1:
                ZH_WORD.build_vocab(train.trg)
            else:
                ZH_WORD.build_vocab(train.trg, max_size=args.tgt_dict_maxSize)
            train_iter, val_iter, test_iter = BucketIterator.splits(
                    (train, val, test), batch_size=args.batch_size, repeat=False)
            return train_iter, val_iter, test_iter, TI_WORD, ZH_WORD
        else:
            exts=(args.extension.split()[0], args.extension.split()[1])
            train, val, test = Trans.splits(path=args.path, exts=exts,
                 fields=(ZH_WORD, EN_WORD), train=args.train, validation=args.valid, test=args.test)
            if args.seg_dict_maxSize == -1:
                ZH_WORD.build_vocab(train.src)
            else:
                ZH_WORD.build_vocab(train.src, max_size=args.seg_dict_maxSize)
            if args.tgt_dict_maxSize == -1:
                EN_WORD.build_vocab(train.trg)
            else:
                EN_WORD.build_vocab(train.trg, max_size=args.tgt_dict_maxSize)

            train_iter, val_iter, test_iter = BucketIterator.splits(
                    (train, val, test), batch_size=args.batch_size, repeat=False)
            return train_iter, val_iter, test_iter, ZH_WORD, EN_WORD

    elif args.mode == 'nmt_char':
        if args.extension.split()[0] == '.ti':
            exts=(args.extension.split()[0], args.extension.split()[1])
            train, val, test = Trans.splits(path=args.path, exts=exts,
                fields=(TI_CHA, ZH_WORD), train=args.train, validation=args.valid, test=args.test)
            if args.src_dict_maxSize == -1:
                TI_CHA.build_vocab(train.src)
            else:
                TI_CHA.build_vocab(train.src, max_size=args.src_dict_maxSize)
            if args.tgt_dict_maxSize == -1:
                ZH_WORD.build_vocab(train.trg)
            else:
                ZH_WORD.build_vocab(train.trg, max_size=args.tgt_dict_maxSize)
            
            train_iter, val_iter, test_iter = BucketIterator.splits(
                    (train, val, test), batch_size=args.batch_size, repeat=False)
            return train_iter, val_iter, test_iter, TI_CHA, ZH_WORD
        else:
            exts=(args.extension.split()[0], args.extension.split()[1])
            train, val, test = Trans.splits(path=args.path, exts=exts,
                fields=(ZH_CHA, EN_WORD), train=args.train, validation=args.valid, test=args.test)
            if args.src_dict_maxSize == -1:
                ZH_CHA.build_vocab(train.src)
            else:
                ZH_CHA.build_vocab(train.src, max_size=args.src_dict_maxSize)
            if args.tgt_dict_maxSize == -1:
                EN_WORD.build_vocab(train.trg)
            else:
                EN_WORD.build_vocab(train.trg, max_size=args.tgt_dict_maxSize)

            train_iter, val_iter, test_iter = BucketIterator.splits(
                    (train, val, test), batch_size=args.batch_size, repeat=False)
            return train_iter, val_iter, test_iter, ZH_CHA, EN_WORD
    
    elif args.mode == 'combine':
        if args.extension.split()[0] == '.ti':
            exts=(args.extension.split()[0], args.extension.split()[1])
            train, val, test = Trans.splits(path=args.path, exts=exts,
                fields=(TI_CHA, ZH_WORD), train=args.train, validation=args.valid, test=args.test)
            if args.src_dict_maxSize == -1:
                TI_CHA.build_vocab(train.src)
            else:
                TI_CHA.build_vocab(train.src, max_size=args.src_dict_maxSize)
            if args.tgt_dict_maxSize == -1:
                ZH_WORD.build_vocab(train.trg)
            else:
                ZH_WORD.build_vocab(train.trg, max_size=args.tgt_dict_maxSize)
            train_iter, val_iter, test_iter = BucketIterator.splits(
                    (train, val, test), batch_size=args.batch_size, repeat=False)
            return train_iter, val_iter, test_iter, TI_CHA, ZH_WORD
        else:
            exts=(args.extension.split()[0], args.extension.split()[1])
            train, val, test = Trans.splits(path=args.path, exts=exts,
                fields=(ZH_CHA, EN_WORD), train=args.train, validation=args.valid, test=args.test)
            if args.src_dict_maxSize == -1:
                ZH_CHA.build_vocab(train.src)
            else:
                ZH_CHA.build_vocab(train.src, max_size=args.src_dict_maxSize)
            if args.tgt_dict_maxSize == -1:
                EN_WORD.build_vocab(train.trg)
            else:
                EN_WORD.build_vocab(train.trg, max_size=args.tgt_dict_maxSize)
            train_iter, val_iter, test_iter = BucketIterator.splits(
                    (train, val, test), batch_size=args.batch_size, repeat=False)
            return train_iter, val_iter, test_iter, ZH_CHA, EN_WORD

    elif args.mode == 'refine_ctc':
        if args.extension.split()[0] == '.ti':
            exts=(args.extension.split()[0], args.extension.split()[1])
            train, val, test = Trans.splits(path=args.path, exts=exts,
                fields=(TI_CHA, TI_WORD), train=args.train, validation=args.valid, test=args.test)
            if args.src_dict_maxSize == -1:
                TI_CHA.build_vocab(train.src)
            else:
                TI_CHA.build_vocab(train.src, max_size=args.src_dict_maxSize)
            if args.seg_dict_maxSize == -1:
                TI_WORD.build_vocab(train.trg)
            else:
                TI_WORD.build_vocab(train.trg, max_size=args.seg_dict_maxSize)
            train_iter, val_iter, test_iter = BucketIterator.splits(
                    (train, val, test), batch_size=args.batch_size, repeat=False)
            return train_iter, val_iter, test_iter, TI_CHA, TI_WORD
        else:
            exts=(args.extension.split()[0], args.extension.split()[1])
            train, val, test = Trans.splits(path=args.path, exts=exts,
                fields=(ZH_CHA, ZH_WORD), train=args.train, validation=args.valid, test=args.test)
            if args.src_dict_maxSize == -1:
                ZH_CHA.build_vocab(train.src)
            else:
                ZH_CHA.build_vocab(train.src, max_size=args.src_dict_maxSize)
            if args.seg_dict_maxSize == -1:
                ZH_WORD.build_vocab(train.trg)
            else:
                ZH_WORD.build_vocab(train.trg, max_size=args.seg_dict_maxSize)
        
            train_iter, val_iter, test_iter = BucketIterator.splits(
                    (train, val, test), batch_size=args.batch_size, repeat=False)
            return train_iter, val_iter, test_iter, ZH_CHA, ZH_WORD

    elif args.mode == 'update_twoLoss':
        if args.extension.split()[0] == '.ti':
            exts = (args.extension.split()[0], args.extension.split()[1], args.extension.split()[2])
            train, val, test, = mydataset.splits(path=args.path, exts=exts, fields=(TI_CHA, ZH_WORD, TI_WORD), train=args.train, validation=args.valid, test=args.test)
            if args.src_dict_maxSize == -1:
                TI_CHA.build_vocab(train.src)
            else:
                TI_CHA.build_vocab(train.src, max_size=args.src_dict_maxSize)
            if args.tgt_dict_maxSize == -1:
                ZH_WORD.build_vocab(train.trg)
            else:
                ZH_WORD.build_vocab(train.trg, max_size=args.tgt_dict_maxSize)
            if args.seg_dict_maxSize == -1:
                TI_WORD.build_vocab(train.ctc)
            else:
                TI_WORD.build_vocab(train.ctc, max_size=args.seg_dict_maxSize)
            train_iter, val_iter, test_iter = BucketIterator.splits(
                    (train, val, test), batch_size=args.batch_size, repeat=False)
            return train_iter, val_iter, test_iter, TI_CHA, ZH_WORD, TI_WORD
        else:
            exts = (args.extension.split()[0], args.extension.split()[1], args.extension.split()[2])
            train, val, test, = mydataset.splits(path=args.path, exts=exts, fields=(ZH_CHA, EN_WORD, ZH_WORD), train=args.train, validation=args.valid, test=args.test)
            if args.src_dict_maxSize == -1:
                ZH_CHA.build_vocab(train.src)
            else:
                ZH_CHA.build_vocab(train.src, max_size=args.src_dict_maxSize)
            if args.tgt_dict_maxSize == -1:
                EN_WORD.build_vocab(train.trg)
            else:
                EN_WORD.build_vocab(train.trg, max_size=args.tgt_dict_maxSize)
            if args.seg_dict_maxSize == -1:
                ZH_WORD.build_vocab(train.ctc)
            else:
                ZH_WORD.build_vocab(train.ctc, max_size=args.seg_dict_maxSize)
            train_iter, val_iter, test_iter = BucketIterator.splits(
                    (train, val, test), batch_size=args.batch_size, repeat=False)

            return train_iter, val_iter, test_iter, ZH_CHA, EN_WORD, ZH_WORD
