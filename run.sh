#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py  -mode nmt_char -epochs 100 -batch_size 1 -lr 0.00001 -grad_clip 1.0 -path '../../../data/AI_chanllenger2017/data_1000k_char/' -extension '.zh .en'  -train 'train' -valid 'val' -test 'test' -embed_size 256 -hidden_size 512 -nmt_encoder_layers 2 -nmt_decoder_layers 1 -ctc_layers 2 -dropout 0.3 -nmt_result_file 'test_nmt_result.txt' -save_path 'checkpoints/charWord_zhen_exp2' -log_FileName 'checkpoints/log_exp2'  -model_cell 'GRU' 



#CUDA_VISIBLE_DEVICES=3 python train.py  -mode ctc  -epochs 100 -batch_size 16 -lr 0.00001 -grad_clip 1.0 -path '../../../data/AI_chanllenger2017/data_1000k_char/' -extension '.zh .seg'  -train 'train' -valid 'val' -test 'test' -embed_size 256 -hidden_size 512 -nmt_encoder_layers 2 -nmt_decoder_layers 1 -ctc_layers 2 -dropout 0.3 -nmt_result_file 'test_nmt_result.txt' -save_path 'checkpoints/test' -log_FileName 'checkpoints/log_test'  -model_cell 'GRU'

#CUDA_VISIBLE_DEVICES='0' python train.py -mode update_twoLoss -epochs 100 -batch_size 12 -lr 0.00001 -grad_clip 10.0 -path '/vision_data/wanglihui/AI_chanllenger2017/data_1000k_char/' -extension '.zh .en .seg' -train 'train' -valid 'val' -test 'test' -nmt_result_file 'test_nmt_result.txt' -seg_result_file 'test_ctc_result.txt' -init_CTC_model 'init_CTCModel/init_ctcModel_6.pt'
