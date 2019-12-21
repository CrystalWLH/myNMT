#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
    继承data.Dataset类，实现可以读取一种输入文本，两种类别监督信息的dataset类
    Author: Lihui Wang     
    Create Date: 2019-03-20
''' 
import os
import io
import re
import pdb
import torchtext.data as data

class MyDataSet(data.Dataset):
    """Defines a dataset for one input and two output labels."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields, **kwargs):
        """Create a MyDataSet given paths and fields."""

        """Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1]), ('ctc', fields[2])]

        src_path, trg_path, ctc_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []

        with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
                io.open(trg_path, mode='r', encoding='utf-8') as trg_file, \
                io.open(ctc_path, mode='r', encoding='utf-8') as ctc_file:
            for src_line, trg_line, ctc_line in zip(src_file, trg_file, ctc_file):
                src_line, trg_line, ctc_line = src_line.strip(), trg_line.strip(), ctc_line.strip()
                if src_line != '' and trg_line != '' and ctc_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line, ctc_line], fields))

        super(MyDataSet, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, exts, fields, path=None, root='.data',
                train='train', validation='val', test='test', **kwargs):
    
        """Create dataset objects for splits of a MyDataset.
            Arguments:
                exts: A tuple containing the extension to path for each language.
                fields: A tuple containing the fields that will be used for data
                    in each language.
                path (str): Common prefix of the splits' file paths, or None to use
                    the result of cls.download(root).
                root: Root dataset storage directory. Default is '.data'.
                train: The prefix of the train data. Default: 'train'.
                validation: The prefix of the validation data. Default: 'val'.
                test: The prefix of the test data. Default: 'test'.
                Remaining keyword arguments: Passed to the splits method of Dataset.
        """
    
        if path is None:
            path = cls.download(root)

        train_data = None if train is None else cls(os.path.join(path, train), exts, fields, **kwargs)
        val_data = None if validation is None else cls(os.path.join(path, validation), exts, fields, **kwargs)
        test_data = None if test is None else cls(os.path.join(path, test), exts, fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data) if d is not None)
    
