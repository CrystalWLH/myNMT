#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
    build model的文件
    Author: Lihui Wang     
    Date: 2019-02-25
''' 

import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.5, model_cell='GRU'):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.model_cell = model_cell
        self.embed = nn.Embedding(input_size, embed_size)
        if model_cell == 'GRU':
            self.model = nn.GRU(embed_size, hidden_size, n_layers,
                            dropout=dropout, bidirectional=True)
        else:
            self.model = nn.LSTM(embed_size, hidden_size, n_layers,
                            dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        embedded = self.embed(src)
        outputs, hidden = self.model(embedded, hidden)
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs, hidden

class Encoder_Combine(nn.Module):
    def __init__(self, input_size, hidden_size,
                 n_layers=1, dropout=0.5, model_cell='GRU'):
        super(Encoder_Combine, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.model_cell = model_cell
        if model_cell == 'GRU':
            self.model = nn.GRU(input_size, hidden_size, n_layers,
                            dropout=dropout, bidirectional=True)
        else:
            self.model = nn.LSTM(input_size, hidden_size, n_layers,
                            dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        outputs, hidden = self.model(src, hidden)
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.relu(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.softmax(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2, model_cell='GRU'):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.model_cell = model_cell

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        if model_cell == 'GRU':
            self.model = nn.GRU(hidden_size + embed_size, hidden_size,
                            n_layers, dropout=dropout)
        else:
            self.model = nn.LSTM(hidden_size + embed_size, hidden_size,
                            n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        if len(last_hidden) == 2:
            attn_weights = self.attention(last_hidden[0][-1], encoder_outputs)
        else:
            attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.model(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights

class CTC_Seg(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, 
                 n_layers=1, dropout=0.5, model_cell='GRU'):
        super(CTC_Seg, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size)
        self.model_cell = model_cell
        if model_cell == 'GRU':
            self.model = nn.GRU(embed_size, hidden_size, n_layers,
                             dropout=dropout, bidirectional=True)
        else:
            self.model = nn.LSTM(embed_size, hidden_size, n_layers,
                            dropout=dropout, bidirectional=True)

        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, cha, word, is_Final=True, hidden=None):
        embedded = self.embed(cha)
        outputs, hidden = self.model(embedded, hidden)
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        if is_Final:
            outputs = self.out(outputs)
        return outputs, hidden


class Seq2Seq(nn.Module):
    def __init__(self, mode, encoder, decoder, ctc_seg=None):
        super(Seq2Seq, self).__init__()
        self.mode = mode
        self.encoder = encoder
        self.decoder = decoder
        self.ctc_seg = ctc_seg

    def forward(self, src, trg, teacher_forcing_ratio=0.5, is_twoLoss_ctc=False):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

        if self.mode == 'ctc':
            seg_output, hidden_seg = self.ctc_seg(src, trg, is_Final=True)
            return seg_output, hidden_seg
        elif self.mode == 'nmt' or self.mode == 'nmt_char':
            encoder_output, hidden_temp = self.encoder(src)
            #判断是GRU还是LSTM的model cell
            #if len(hidden_temp) == 2:
            #    hidden = hidden_temp[0]
            #else:
            #    hidden = hidden_temp[0]
            
            #hidden = hidden[:self.decoder.n_layers]
            hidden = (hidden_temp[0][:self.decoder.n_layers], hidden_temp[1][:self.decoder.n_layers])

            output = Variable(trg.data[0, :])  # sos
            for t in range(1, max_len):
                output, hidden, attn_weights = self.decoder(
                        output, hidden, encoder_output)
                outputs[t] = output
                #判断是GRU还是LSTM的model cell
                #if len(hidden_temp) == 2:
                #    hidden = hidden_temp[0]
                #else:
                #    hidden = hidden_temp
                is_teacher = random.random() < teacher_forcing_ratio
                top1 = output.data.max(1)[1]
                output = Variable(trg.data[t] if is_teacher else top1).cuda()
            return outputs
        elif self.mode == 'combine':
            seg_output, hidden_seg_temp = self.ctc_seg(src, trg, is_Final=False)
            #判断是GRU还是LSTM的model cell
            if len(hidden_seg_temp) == 2:
                hidden_seg = hidden_seg_temp[0]
            else:
                hidden_seg = hidden_seg_temp
            encoder_output, hidden_temp = self.encoder(seg_output, hidden=hidden_seg)
            #判断是GRU还是LSTM的model cell
            if len(hidden_temp) == 2:
                hidden = hidden_temp[0]
            else:
                hidden = hidden_temp
            hidden = hidden[:self.decoder.n_layers]
            output = Variable(trg.data[0, :])  # sos
            for t in range(1, max_len):
                output, hidden_temp, attn_weights = self.decoder(
                        output, hidden, encoder_output)
                outputs[t] = output
                #判断是GRU还是LSTM的model cell
                if len(hidden_temp) == 2:
                    hidden = hidden_temp[0]
                else:
                    hidden = hidden_temp
                is_teacher = random.random() < teacher_forcing_ratio
                top1 = output.data.max(1)[1]
                output = Variable(trg.data[t] if is_teacher else top1).cuda()
            return outputs, seg_output

        elif self.mode == 'refine_ctc':
            seg_output, hidden_seg_temp = self.ctc_seg(src, trg, is_Final=True)
            #判断是GRU还是LSTM的model cell
            if len(hidden_seg_temp) == 2:
                hidden_seg = hidden_seg_temp[0]
            else:
                hidden_seg = hidden_seg_temp
            return seg_output, hidden_seg

        elif self.mode == 'update_twoLoss':
            if is_twoLoss_ctc:
                seg_output, hidden_seg_temp = self.ctc_seg(src, trg, is_Final=True)
                #判断是GRU还是LSTM的model cell
                if len(hidden_seg_temp) == 2:
                    hidden_seg = hidden_seg_temp[0]
                else:
                    hidden_seg = hidden_seg_temp
                return seg_output, hidden_seg
            else:
                seg_output, hidden_seg_temp = self.ctc_seg(src, trg, is_Final=False)
                #判断是GRU还是LSTM的model cell
                if len(hidden_seg_temp) == 2:
                    hidden_seg = hidden_seg_temp[0]
                else:
                    hidden_seg = hidden_seg_temp

                encoder_output, hidden_temp = self.encoder(seg_output, hidden=hidden_seg)
                #判断是GRU还是LSTM的model cell
                if len(hidden_temp) == 2:
                    hidden = hidden_temp[0]
                else:
                    hidden = hidden_temp

                hidden = hidden[:self.decoder.n_layers]
                output = Variable(trg.data[0, :])  # sos
                for t in range(1, max_len):
                    output, hidden_temp, attn_weights = self.decoder(
                            output, hidden, encoder_output)
                    outputs[t] = output
                    #判断是GRU还是LSTM的model cell
                    if len(hidden_temp) == 2:
                        hidden = hidden_temp[0]
                    else:
                        hidden = hidden_temp
                    is_teacher = random.random() < teacher_forcing_ratio
                    top1 = output.data.max(1)[1]
                    output = Variable(trg.data[t] if is_teacher else top1).cuda()
                return outputs, seg_output
        else:
            print('Please input correct training mode. Such as: ctc | nmt | nmt_char | combine | refine_ctc | update_twoLoss')
            return
