#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python train.py  -mode nmt_char -epochs 100 -batch_size 8 -lr 0.00001 -grad_clip 1.0 -path '../../../data/AI_chanllenger2017/data_1000k_char/' -extension '.zh .en'  -train 'train' -valid 'val' -test 'test' -embed_size 256 -hidden_size 512 -nmt_encoder_layers 2 -nmt_decoder_layers 1 -ctc_layers 2 -dropout 0.3 -nmt_result_file 'checkpoints/test_temp.txt' -val_nmt_result_file 'checkpoints/val_temp.txt' -save_path 'checkpoints/temp' -log_FileName 'checkpoints/log_temp'  -model_cell 'GRU' -tgt_dict_maxSize 50000 -model_path 'checkpoints/charWord_zhen_exp2/seq2seq_5.pt' -istest

#CUDA_VISIBLE_DEVICES=2 python train.py  -mode nmt_char -epochs 100 -batch_size 16 -lr 0.00001 -grad_clip 1.0 -path '../../../data/AI_chanllenger2017/data_1000k_char/' -extension '.zh .en'  -train 'train' -valid 'val' -test 'test' -embed_size 256 -hidden_size 512 -nmt_encoder_layers 2 -nmt_decoder_layers 1 -ctc_layers 2 -dropout 0.3 -nmt_result_file 'checkpoints/test_nmt_result_exp1.txt' -val_nmt_result_file 'checkpoints/val_nmt_result_exp1.txt'  -save_path 'checkpoints/charWord_zhen_exp1' -log_FileName 'checkpoints/log_exp1'  -model_cell 'LSTM' --tgt_dict_maxSize 50000

#CUDA_VISIBLE_DEVICES=2 python train.py  -mode nmt_char -epochs 100 -batch_size 8 -lr 0.00001 -grad_clip 1.0 -path '../../../data/AI_chanllenger2017/data_1000k_char/' -extension '.zh .en'  -train 'train' -valid 'val' -test 'test' -embed_size 256 -hidden_size 512 -nmt_encoder_layers 2 -nmt_decoder_layers 1 -ctc_layers 2 -dropout 0.3 -nmt_result_file 'checkpoints/test_nmt_result_exp2.txt' -val_nmt_result_file 'checkpoints/val_nmt_result_exp2.txt' -save_path 'checkpoints/charWord_zhen_exp2' -log_FileName 'checkpoints/log_exp2'  -model_cell 'GRU' -tgt_dict_maxSize 50000

#CUDA_VISIBLE_DEVICES=2 python train.py  -mode nmt_char -epochs 100 -batch_size 16 -lr 0.00001 -grad_clip 1.0 -path '../../../data/AI_chanllenger2017/data_1000k_char/' -extension '.zh .en'  -train 'train' -valid 'val' -test 'test' -embed_size 256 -hidden_size 512 -nmt_encoder_layers 2 -nmt_decoder_layers 2 -ctc_layers 2 -dropout 0.3 -nmt_result_file 'checkpoints/test_nmt_result_exp3.txt' -val_nmt_result_file 'checkpoints/val_nmt_result_exp3.txt'  -save_path 'checkpoints/charWord_zhen_exp3' -log_FileName 'checkpoints/log_exp3'  -model_cell 'LSTM' -tgt_dict_maxSize 50000

#CUDA_VISIBLE_DEVICES=0 python train.py  -mode nmt_char -epochs 100 -batch_size 8 -lr 0.00001 -grad_clip 1.0 -path '../../../data/AI_chanllenger2017/data_1000k_char/' -extension '.zh .en'  -train 'train' -valid 'val' -test 'test' -embed_size 256 -hidden_size 512 -nmt_encoder_layers 2 -nmt_decoder_layers 2 -ctc_layers 2 -dropout 0.3 -nmt_result_file 'checkpoints/test_nmt_result_exp4.txt' -val_nmt_result_file 'checkpoints/val_nmt_result_exp4.txt' -save_path 'checkpoints/charWord_zhen_exp4' -log_FileName 'checkpoints/log_exp4'  -model_cell 'GRU' -tgt_dict_maxSize 50000






