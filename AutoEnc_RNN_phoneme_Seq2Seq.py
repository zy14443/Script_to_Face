# -*- coding: utf-8 -*-

import sys
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time
import math
import random
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

from fairseq import options, utils
from fairseq.modules import (
    AdaptiveSoftmax, BeamableMM, GradMultiply, LearnedPositionalEmbedding,
    LinearizedConvolution,
)

PAD_IDX = 0
MAX_LENGTH = 40
EOS_IDX = 85
N_ID=3744

def make_mlp(dim_list, activation='relu', batch_norm=False, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)

class RNN_Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, RNN_type, device, num_layers=1, dropout=0.0):
        super(RNN_Encoder, self).__init__()
        self.enc_hid_dim = enc_hid_dim
        self.num_layers = num_layers        
        self.RNN_type = RNN_type              

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, enc_hid_dim, num_layers, dropout=dropout)
        self.lstm = nn.LSTM(emb_dim, enc_hid_dim, num_layers, dropout=dropout)
        self.device = device
        

    def forward(self, input, hidden, input_len):
        embedded = self.embedding(input)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_len)        
            
        if self.RNN_type == 'GRU':
            packed_outputs, hidden = self.gru(packed_embedded, hidden)
        elif self.RNN_type == 'LSTM':
            packed_outputs, hidden = self.lstm(packed_embedded, hidden)
        
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        
        return outputs, hidden

    def initHidden(self,batch_size):
        if self.RNN_type == 'GRU':
            return torch.zeros(self.num_layers, batch_size, self.enc_hid_dim, device=self.device)
        elif self.RNN_type == 'LSTM':
            return (torch.zeros(self.num_layers, batch_size, self.enc_hid_dim, device=self.device),
                    torch.zeros(self.num_layers, batch_size, self.enc_hid_dim, device=self.device)
                   )

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        self.attn = nn.Linear((enc_hid_dim) + dec_hid_dim, dec_hid_dim)
#        self.attn = nn.Linear((enc_hid_dim) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))
    
    def forward(self, hidden, encoder_outputs, mask):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2))) 
        energy = energy.permute(0, 2, 1)
        
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention = torch.bmm(v, energy).squeeze(1)
        mask = mask[:,range(encoder_outputs.shape[1])]
        attention = attention.masked_fill(mask == 0, float('-inf'))
        
        return F.softmax(attention, dim=1)

class RNN_Decoder_1(nn.Module):
    def __init__(self, dec_hid_dim, output_dim, enc_hid_dim, attention, dropout, RNN_type, device, num_layers=1):
        super(RNN_Decoder_1, self).__init__()
        self.RNN_type = RNN_type  
        self.dec_hid_dim = dec_hid_dim
        self.enc_hid_dim = enc_hid_dim  
        self.num_layers = num_layers
        self.attention = attention
        
        self.gru = nn.GRU(enc_hid_dim + output_dim, dec_hid_dim, num_layers, dropout=dropout)
        self.lstm = nn.LSTM(enc_hid_dim + output_dim, dec_hid_dim, num_layers, dropout=dropout)        
        
#        self.out_decoder = make_mlp([enc_hid_dim + dec_hid_dim + output_dim+128,256,128], activation='relu', batch_norm=False, dropout=0)
        
        self.out = nn.Linear(enc_hid_dim + dec_hid_dim + output_dim+128, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.device = device

    def forward(self, input, hidden, encoder_outputs, mask, cell_id):
        input = input.unsqueeze(0)
        if self.RNN_type == 'GRU':
            squeeze_hidden = hidden[self.num_layers-1]
            a = self.attention(squeeze_hidden,encoder_outputs, mask)            
        elif self.RNN_type == 'LSTM':
            squeeze_hidden = (hidden[0][self.num_layers-1], hidden[1][self.num_layers-1])
            a = self.attention(squeeze_hidden[0],encoder_outputs,  mask)            
                
        a = a.unsqueeze(1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        
        weighted = weighted.permute(1, 0, 2)
        
        rnn_input = torch.cat((input, weighted), dim=2)
        
        if self.RNN_type == 'GRU':
            output, hidden = self.gru(rnn_input, hidden)
        elif self.RNN_type == 'LSTM':
            output, hidden = self.lstm(rnn_input, hidden)

        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        input = input.squeeze(0)
        
#        output = self.out_decoder(torch.cat((output, weighted, input,cell_id), dim=1))         
        output = self.sigmoid(self.out(torch.cat((output, weighted, input,cell_id), dim=1)))

        return output, hidden

    def initHidden(self, batch_size):
        if self.RNN_type == 'GRU':
            return torch.zeros(self.num_layers, batch_size, self.enc_hid_dim, device=self.device)
        elif self.RNN_type == 'LSTM':
            return (torch.zeros(self.num_layers, batch_size, self.enc_hid_dim, device=self.device),
                    torch.zeros(self.num_layers, batch_size, self.enc_hid_dim, device=self.device)
                   )

class RNN_Decoder_2(nn.Module):
    def __init__(self, dec_hid_dim, output_dim, enc_hid_dim, attention, dropout, RNN_type, device, num_layers=1):
        super(RNN_Decoder_2, self).__init__()
        self.RNN_type = RNN_type  
        self.dec_hid_dim = dec_hid_dim
        self.enc_hid_dim = enc_hid_dim  
        self.num_layers = num_layers
        self.attention = attention
        
        self.gru = nn.GRU(128, dec_hid_dim, num_layers, dropout=dropout)
        self.lstm = nn.LSTM(128, dec_hid_dim, num_layers, dropout=dropout)        
        
        self.embedding = nn.Embedding(output_dim, 128)
        
#        self.input_encoder = make_mlp([output_dim,128,128], activation='relu', batch_norm=False, dropout=0)        
        self.out_decoder = make_mlp([dec_hid_dim,256], activation='relu', batch_norm=False, dropout=0)        
        self.out = nn.Linear(256, output_dim)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.device = device

    def forward(self, input, hidden, encoder_outputs):
#        input = input.unsqueeze(0)
#        if self.RNN_type == 'GRU':
#            squeeze_hidden = hidden[self.num_layers-1]
#            a = self.attention(squeeze_hidden,encoder_outputs, mask)            
#        elif self.RNN_type == 'LSTM':
#            squeeze_hidden = (hidden[0][self.num_layers-1], hidden[1][self.num_layers-1])
#            a = self.attention(squeeze_hidden[0],encoder_outputs,  mask)            
#                
#        a = a.unsqueeze(1)
#        
#        encoder_outputs = encoder_outputs.permute(1, 0, 2)
#        weighted = torch.bmm(a, encoder_outputs)
#        
#        weighted = weighted.permute(1, 0, 2)
        
#        rnn_input = torch.cat((input, weighted), dim=2) #(L,B,N)
#        rnn_input = input.squeeze(0)
#        rnn_input = self.input_encoder(rnn_input)
#        rnn_input = rnn_input.unsqueeze(0)
        embedded = self.embedding(input)

        
        if self.RNN_type == 'GRU':
            output, hidden = self.gru(embedded, hidden)
        elif self.RNN_type == 'LSTM':
            output, hidden = self.lstm(embedded, hidden)

        output = output.squeeze(0)
#        weighted = weighted.squeeze(0)
#        input = input.squeeze(0)
        
#        output = self.out_decoder(torch.cat((output, weighted, input,cell_id), dim=1))
        output = self.out_decoder(output)                 
        output = self.out(output)

        return output, hidden

    def initHidden(self, batch_size):
        if self.RNN_type == 'GRU':
            return torch.zeros(self.num_layers, batch_size, self.enc_hid_dim, device=self.device)
        elif self.RNN_type == 'LSTM':
            return (torch.zeros(self.num_layers, batch_size, self.enc_hid_dim, device=self.device),
                    torch.zeros(self.num_layers, batch_size, self.enc_hid_dim, device=self.device)
                   )
           
class Seq2Seq_Generator_1(nn.Module):
    def __init__ (self, encoder, decoder, device):
        super(Seq2Seq_Generator_1, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.embedding = nn.Linear(128, self.encoder.num_layers*self.encoder.enc_hid_dim)
    
    def create_mask(self, src):
        mask = (src != PAD_IDX).permute(1, 0)
        return mask 
    
    def forward(self, input, input_len, target, p_id, teacher_forcing_ratio=0.5):
        
        batch_size = input.shape[0]
        target_size = 1
        max_len = target.shape[1]
        
        input = input.permute(1,0)
        target = target.permute(1,0)
                
        outputs = torch.zeros(MAX_LENGTH, batch_size,41).to(self.device)
        
        encoder_hidden = self.encoder.initHidden(batch_size)
        encoder_outputs, hidden = self.encoder(input, encoder_hidden, input_len)
        
#        encoder_hidden[0].shape
#       
#        p_id = p_id.squeeze()
#        if batch_size==1:
#            cell_id = p_id.unsqueeze(0)
#        else:
#            cell_id = p_id
#        cell_id = self.embedding(p_id)
#        if self.encoder.RNN_type == 'LSTM':
#            cell_id = cell_id.view(encoder_hidden[0].shape[1],encoder_hidden[0].shape[0],-1)
#            cell_id = cell_id.permute(1,0,2).contiguous()
#            new_hidden = (hidden[0],cell_id)
#        elif self.encoder.RNN_type == 'GRU':
#            cell_id = cell_id.view(encoder_hidden.shape[1],encoder_hidden.shape[0],-1)
#            cell_id = cell_id.permute(1,0,2).contiguous()
#            new_hidden = torch.cat((hidden,cell_id),dim=2)
        new_hidden = hidden
#        cell_noise = get_noise(encoder_hidden[0].shape,'uniform').to(self.device)
        
        
        output = torch.zeros(batch_size, target_size).to(self.device)
        output = output.type(torch.long)
        output = output.permute(1,0)
#        mask = self.create_mask(input)
        
        for t in range(0, max_len):
            output, new_hidden = self.decoder(output, new_hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio    
            top1 = output.max(1)[1]
            top1 = top1.unsqueeze(0)
            output = (target[t] if teacher_force else top1)

        return outputs

class Seq2Seq_Generator_2(nn.Module):
    def __init__ (self, encoder, decoder, device):
        super(Seq2Seq_Generator_2, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.embedding = nn.Linear(128, self.encoder.num_layers*self.encoder.enc_hid_dim)
    
    def create_mask(self, src):
        mask = (src != PAD_IDX).permute(1, 0)
        return mask 
    
    def forward(self, input, input_len, target, p_id, teacher_forcing_ratio=0.5):
        
        batch_size = input.shape[0]
        target_size = target.shape[2]
        max_len = target.shape[1]
        
        input = input.permute(1,0)
        target = target.permute(1,0,2)
                
        outputs = torch.zeros(MAX_LENGTH, batch_size, target_size).to(self.device)
        
        encoder_hidden = self.encoder.initHidden(batch_size)
        encoder_outputs, hidden = self.encoder(input, encoder_hidden, input_len)
        
#        encoder_hidden[0].shape
#       
        p_id = p_id.squeeze()
#        cell_id = p_id.unsqueeze(0)
        cell_id = self.embedding(p_id)
        if self.encoder.RNN_type == 'LSTM':
            cell_id = cell_id.view(encoder_hidden[0].shape[1],encoder_hidden[0].shape[0],-1)
            cell_id = cell_id.permute(1,0,2).contiguous()
            new_hidden = (hidden[0],cell_id)
        elif self.encoder.RNN_type == 'GRU':
            cell_id = cell_id.view(encoder_hidden.shape[1],encoder_hidden.shape[0],-1)
            cell_id = cell_id.permute(1,0,2).contiguous()
            new_hidden = torch.cat((hidden,cell_id),dim=2)
#        new_hidden = hidden
#        cell_noise = get_noise(encoder_hidden[0].shape,'uniform').to(self.device)
        if batch_size==1:
            p_cell_id = p_id.unsqueeze(0)
        else:
            p_cell_id = p_id
#        p_cell_id = p_id.unsqueeze(0)
        
        output = torch.zeros(batch_size, target_size).to(self.device)        
        mask = self.create_mask(input)
        
        for t in range(0, max_len):
            output, new_hidden = self.decoder(output, new_hidden, encoder_outputs, mask,p_cell_id)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio    
            
            
            output = (target[t] if teacher_force else output)

        return outputs              