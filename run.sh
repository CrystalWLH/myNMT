#!/bin/bash

CUDA_VISIBLE_DEVICES='1' python train.py -mode nmt_char -epochs 100 -batch_size 12 -lr 0.00001 -grad_clip 10.0 -path '/home/wanglh/data_language/CCMT/charData/' -extension '.ti .zh' -train 'train' -valid 'valid' -test 'test_2017_0' -nmt_result_file 'test_nmt_result.txt' -hidden_size 256 -embedding_size 256 -ctc_layers 2 -encoder_layers 2 -decoder_layers 1
