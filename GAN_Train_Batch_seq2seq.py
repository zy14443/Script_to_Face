#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 12:15:49 2018

@author: zheng.1443
"""
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

import util_func
import au_dataloader
import loss_func
import model_Word2Vec_RNN
import model_RNN_Seq2Seq
import model_CNN_Seq2Seq


#%%
if sys.version_info[0] < 3:
    f_handler = open('./phone_data/word_cmu.pkl','r')
    [word_cmu, word_nouse] = pickle.load(f_handler)
    f_handler.close()
    
    f_handler = open('./phone_data/word_to_phone_symbol.pkl','r')
    [word_to_phone, word_to_symbol] = pickle.load(f_handler)
    f_handler.close()

#f_handler = open('word_to_phone_vector.pkl','r')
#[transformed] = pickle.load(f_handler)
#f_handler.close()

else:    
    f_handler = open('./phone_data/word_cmu.pkl','rb')
    [word_cmu, word_nouse] = pickle.load(f_handler,encoding='latin1')
    f_handler.close()
    
    f_handler = open('./phone_data/word_to_phone_symbol.pkl','rb')
    [word_to_phone, word_to_symbol] = pickle.load(f_handler, encoding='latin1')
    f_handler.close()

#
#f_handler = open('word_to_phone_vector.pkl','rb')
#[transformed] = pickle.load(f_handler, encoding='latin1')
#f_handler.close()

w2i = {w: i for i, w in enumerate(word_cmu)}
i2w = {i: w for i, w in enumerate(word_cmu)}

#mode_AUs = sio.loadmat(file_name)
opt = [];

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = "cpu"


generator_models = ('Word2Vec','Seq2Seq','CNN')
model_selection = generator_models[1]

#%% Dataset

#file_names = glob.glob("../train_AUs_single_reduced_len/*.mat")
file_names = sorted(glob.glob("/eecf/cbcsl/data100b/zheng.1443/Data/Word_Frames/train_AUs_LRW_mini/*.mat"))
#file_names_val = sorted(glob.glob("/eecf/cbcsl/data100b/zheng.1443/Data/Word_Frames/train_AUs_LRW/val/*.mat"))

mini_batch_size = 16
NUM_AUs = 16
PAD_IDX = 0
MAX_LENGTH = 40
EOS_IDX = 40

au_dataloader_list = []
for i in range(len(file_names)):
    t_dataset = au_dataloader.word_AU_Dataset(opt,file_names[i], w2i, word_to_symbol)
    t_dataloader = DataLoader(t_dataset, batch_size=mini_batch_size, shuffle=True, drop_last=True)
    au_dataloader_list.append(t_dataloader)

#au_dataloader_list_val = []
#for i in range(len(file_names_val)):
#    v_dataset = au_dataloader.word_AU_Dataset_val(opt,file_names_val[i], w2i, word_to_symbol)
#    v_dataloader = DataLoader(v_dataset, batch_size=1, shuffle=False, drop_last=True)
#    au_dataloader_list_val.append(v_dataloader)

# Check if the data is loaded correctly
#
#for i, sample in enumerate(au_dataloader_list_val[12]):
#    print(i2w[sample['word'].item()])
#    if i==2:
#        break
    
for i, sample in enumerate(au_dataloader_list[1]):
    print(sample['phoneme_len'])
    if i==2:
        break


#%% Neural Network Sturcture
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


class AU_Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, RNN_type, activation='sigmoid', batch_norm=True, dropout=0.0):
        super(AU_Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers        
        self.RNN_type = RNN_type  
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, dropout=dropout)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout)
        
        self.real_classifier = make_mlp([2*hidden_dim,hidden_dim,1], activation=activation, batch_norm=batch_norm, dropout=dropout)

        
    def forward(self, input, hidden, input_len):
        packed_embedded = nn.utils.rnn.pack_padded_sequence(input, input_len)        

        if self.RNN_type == 'GRU':
            packed_outputs, hidden = self.gru(packed_embedded, hidden)
        elif self.RNN_type == 'LSTM':
            packed_outputs, hidden = self.lstm(packed_embedded, hidden)

#        outputs = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        scores = self.real_classifier(hidden[0].view(30,-1))

        return scores
    
    def initHidden(self,batch_size):
        if self.RNN_type == 'GRU':
            return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        elif self.RNN_type == 'LSTM':
            return (torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
                    torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
                   )
            
def train_Discriminator(au_descriminator, output_tensor):
    
    gen_len = np.round(np.random.rand(30)*21+1)
    gen_len = torch.as_tensor([gen_len], dtype = torch.long, device=device)
    
    gen_len, index = torch.sort(gen_len.squeeze(0),descending = True)
    output_tensor = output_tensor[:,index]
    encoder_hidden = au_descriminator.initHidden(30)
    scores= au_descriminator(output_tensor, encoder_hidden, gen_len)
    
    
    return scores

def compute_gradient_penalty(descriminator, real_samples, fake_samples):
    
    real_samples = real_samples.permute(1,0,2)
    fake_samples = fake_samples.permute(1,0,2)
    fake_samples = fake_samples[:,0:real_samples.shape[1]]
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.tensor(np.random.random((real_samples.size(0), 1, 1)),dtype = torch.float32).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    interpolates = interpolates.permute(1,0,2).to(device)
    
    gen_len = np.ones(real_samples.shape[0])*real_samples.shape[1]
    gen_len = torch.as_tensor(gen_len, dtype = torch.long, device=device)
    
    encoder_hidden = descriminator.initHidden(30)
    d_interpolates = descriminator(interpolates, encoder_hidden, gen_len)    
  
    fake = torch.ones(d_interpolates.size(), requires_grad=False).to(device)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(1), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

#%%
if model_selection == 'CNN':
    INPUT_DIM = 86
    OUTPUT_DIM = 16
    
    EMB_DIM = 256
    HID_DIM = 512
    
    ENC_LAYERS = 10
    DEC_LAYERS = 10
    
    ENC_KERNEL_SIZE = 3
    DEC_KERNEL_SIZE = 3
    ENC_DROPOUT = 0.25
    DEC_DROPOUT = 0.25

    enc = model_CNN_Seq2Seq.CNN_Encoder(INPUT_DIM, EMB_DIM).to(device)
    dec = model_CNN_Seq2Seq.CNN_Decoder(EMB_DIM, EMB_DIM, OUTPUT_DIM).to(device)
    
    model = model_CNN_Seq2Seq.CNN_Generator(enc,dec,device).to(device)
    print ('N parameters: %d' % util_func.count_parameters(model))
    
    learning_rate=0.001 
    optimizer_G = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_model = loss_func.loss_cosine_MSE_seq(beta=0.5)

    def train_CNN(model, data_loader, optimizer, criterion):
        model.train()
        
        epoch_loss = 0
        
        for i, batch in enumerate(data_loader):
    
            input_tensor = batch['phoneme'].to(device)
            input_len = batch['phoneme_len'].to(device)
            target_tensor = batch['AU_sequence'].to(device)
                     
            optimizer.zero_grad()
            
            output_tensor = model(input_tensor, input_len, target_tensor)
            output_tensor = output_tensor.permute(1,0,2)
            loss = criterion.loss_fn(output_tensor, target_tensor)
            
            loss.backward()
            
    #        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
        return epoch_loss, len(data_loader)
  
    def evaluate_CNN(model, data_loader, criterion):
        model.eval()
        
        epoch_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                
                input_tensor = batch['phoneme'].to(device)
                input_len = batch['phoneme_len'].to(device)
                target_tensor = batch['AU_sequence'].to(device)
                            
                output_tensor = model(input_tensor, input_len, target_tensor, 0)
                output_tensor = output_tensor.permute(1,0,2)
                loss = criterion_model.loss_fn(output_tensor, target_tensor)
                
                  
                epoch_loss += loss.item()
            
        return epoch_loss, len(data_loader)
    
    n_epoch = 10
    epoch_start = 0
    load_epoch = False
    
    print_every=1
    plot_every=1
    save_every = 1
    
    vocab_size = len(w2i)
    
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every 
    
    print_loss_total_val = 0  # Reset every print_every
    plot_loss_total_val = 0  # Reset every plot_every 
#%%     
    print('epoch_start: %d n_epoch: %d' % (epoch_start+1, n_epoch))
   
    start = time.time()    
    for epoch in range(epoch_start+1,n_epoch+1):
        
        len_index_all = range(len(au_dataloader_list)) 
        n_batches = 0
    #    for len_index in len_index_all:
        len_index=1
        temp_AU_loss = 0
    #    temp_EOS_loss = 0    
    
#        train_loss_D, train_loss_G, dataloader_len = train_GAN_Seq2Seq(au_generator, au_descriminator, au_dataloader_list[len_index], optimizer_G, optimizer_D)    
    
        train_loss, dataloader_len = train_CNN(model, au_dataloader_list[len_index] , optimizer_G, criterion_model)
        
        n_batches += dataloader_len
        temp_AU_loss += train_loss
        
        
        print_loss_total += temp_AU_loss
        plot_loss_total += temp_AU_loss        
           
              
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / (print_every*n_batches)
            print_loss_total = 0
            print('train: (%d %d%%) %.4f' % (epoch, float(epoch-epoch_start) / (n_epoch-epoch_start) * 100, print_loss_avg))
        
        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / (plot_every*n_batches)
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        
        
        n_batches = 0
        temp_AU_loss = 0
        val_loss, dataloader_len = evaluate_CNN(model, au_dataloader_list_val[len_index] , criterion_model)
        n_batches += dataloader_len
        temp_AU_loss += val_loss
        
        print_loss_total_val += temp_AU_loss
        
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total_val / (print_every*n_batches)
            print_loss_total_val = 0
            print('val: (%d %d%%) %.4f' % (epoch, float(epoch-epoch_start) / (n_epoch-epoch_start) * 100, print_loss_avg))
        
        
        print('%s' % (util_func.timeSince(start, float(epoch-epoch_start) / (n_epoch-epoch_start))))        
        
#        if epoch % save_every == 0:
#            torch.save({
#            'epoch': epoch,
#            'model_state_dict': model.state_dict(),
#            'model_optimizer_state_dict': model.state_dict(),
#            'loss': plot_losses,            
#            }, './checkpoints/Seq2Seq_batch_epoch_%d.pth' % epoch)
    
    
#%%
if model_selection == 'Seq2Seq':
    
    INPUT_DIM = 41
    OUTPUT_DIM = 16
    
    ENC_EMB_DIM = 256
    
    ENC_HID_DIM = 256
    DEC_HID_DIM = 256
    
    ENC_DROPOUT = 0
    DEC_DROPOUT = 0
    
    N_LAYER = 2
    
    attn = model_RNN_Seq2Seq.Attention(ENC_HID_DIM, DEC_HID_DIM).to(device)
    enc = model_RNN_Seq2Seq.RNN_Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, 'LSTM', device, N_LAYER, ENC_DROPOUT).to(device)
    dec = model_RNN_Seq2Seq.RNN_Decoder_2(DEC_HID_DIM, OUTPUT_DIM, ENC_HID_DIM, attn, DEC_DROPOUT, 'LSTM', device, N_LAYER).to(device)
    
    au_generator = model_RNN_Seq2Seq.Seq2Seq_Generator_1(enc, dec, device).to(device)   
#    au_descriminator = AU_Discriminator(OUTPUT_DIM, DEC_HID_DIM, N_LAYER, 'LSTM').to(device)
    

    print ('N parameters: %d' % util_func.count_parameters(au_generator))
    
    learning_rate=0.0001 
    
    optimizer_G = optim.Adam(au_generator.parameters(), lr=learning_rate)
#    optimizer_D = optim.Adam(au_descriminator.parameters(), lr=learning_rate)
    
    criterion_model = loss_func.loss_cosine_MSE_seq(beta=0.8)
    
    
    
#    def train_GAN_Seq2Seq(generator, descriminator, data_loader, optimizer_G, optimizer_D):
#        generator.train()
#        descriminator.train()
#        epoch_loss_D = 0
#        epoch_loss_G = 0
#        
#        for i, batch in enumerate(data_loader):
#            
#            input_tensor = batch['phoneme'].to(device)
#            input_len = batch['phoneme_len'].to(device)            
#            target_tensor = batch['AU_sequence'].to(device)
#            
#            input_len, index = torch.sort(input_len,descending = True)
#            input_tensor = input_tensor[index,]
#            target_tensor = target_tensor[index,]
#        
#        
#            optimizer_D.zero_grad()
#        
#            output_tensor = generator(input_tensor, input_len, target_tensor,0)
#            
#            fake_scores = train_Discriminator(descriminator, output_tensor)
#            target_tensor = target_tensor.permute(1,0,2)
#            real_scores = train_Discriminator(descriminator, target_tensor)
#
#            gradient_penalty = compute_gradient_penalty(descriminator, target_tensor, output_tensor)
#            
#        
#            d_loss = -torch.mean(fake_scores) + torch.mean(real_scores) + 10 * gradient_penalty
#            epoch_loss_D += d_loss.item()
#        
#            d_loss.backward()
#            optimizer_D.step()
#        
#
#            optimizer_G.zero_grad()
#            
#            output_tensor = generator(input_tensor, input_len, target_tensor,0)
#            fake_scores = train_Discriminator(descriminator, output_tensor)
#            g_loss = -torch.mean(fake_scores)
#    
#            epoch_loss_G += g_loss.item()
#            
#            g_loss.backward()
#            optimizer_G.step()
#    
#        return epoch_loss_D, epoch_loss_G,  len(data_loader)
    
    
    def train_Seq2Seq(model, data_loader, optimizer, criterion, clip, tf_rate=0.5):
        model.train()
        
        epoch_loss = 0
        
        for i, batch in enumerate(data_loader):
#            for j in range(1,150):
            input_tensor = batch['phoneme'].to(device)
            input_len = batch['phoneme_len'].to(device)
            target_tensor = batch['AU_sequence'].to(device)
            p_id = batch['p_id'].to(device)
            
            input_len, index = torch.sort(input_len,descending = True)
            input_tensor = input_tensor[index,]
            target_tensor = target_tensor[index,]
            
            optimizer.zero_grad()
            
            output_tensor = model(input_tensor, input_len, target_tensor,p_id,tf_rate)
            
            loss = criterion.loss_fn(output_tensor, target_tensor)
#                if j==1:
#                    print('iteration %d start: %f' % (i,loss.item()))
            
            loss.backward()
            
    #        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            optimizer.step()
#            print ('iteration %d end: %f' % (i,loss.item()))    
            epoch_loss += loss.item()
        
        encoder_grad = list()
        for p in list(filter(lambda p: p.grad is not None, model.encoder.parameters())):
            encoder_grad.append(p.grad.data.mean().item())
        print(encoder_grad)
        
        decoder_grad = list()
        for p in list(filter(lambda p: p.grad is not None, model.decoder.parameters())):
            decoder_grad.append(p.grad.data.mean().item())
        print(decoder_grad)
        
        return epoch_loss, len(data_loader)
        
    
    def evaluate_Seq2Seq(model, data_loader, criterion):
        model.eval()
        
        epoch_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                input_tensor = batch['phoneme'].to(device)
                input_len = batch['phoneme_len'].to(device)
                target_tensor = batch['pose_sequence'].to(device)
                p_id = batch['p_id'].to(device)

                
                input_len, index = torch.sort(input_len,descending = True)
                input_tensor = input_tensor[index,]
                target_tensor = target_tensor[index,]
                            
                output_tensor = model(input_tensor, input_len, target_tensor, p_id,0)
                
                loss = criterion.loss_fn(output_tensor, target_tensor)
                
                  
                epoch_loss += loss.item()
            
        return epoch_loss, len(data_loader), output_tensor
    
 
    n_epoch = 100
    epoch_start = 0
    
    print_every=1
    plot_every=1
    save_every = 5
    
    vocab_size = len(w2i)
    
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every 
    
    plot_losses_val = []
    print_loss_total_val = 0  # Reset every print_every
    plot_loss_total_val = 0  # Reset every plot_every 

    load_epoch = False

    if(load_epoch):
        print('test')
        checkpoint = torch.load('./checkpoints/Seq2Seq_batch_LSTM_2_epoch_200.pth',map_location=device)
        plot_losses = checkpoint['train_loss']
        plot_losses_val = checkpoint['val_loss']
        epoch_start = checkpoint['epoch']
        au_generator.load_state_dict(checkpoint['model_state_dict'])
        optimizer_G.load_state_dict(checkpoint['model_optimizer_state_dict'])

#%%    
    print('epoch_start: %d n_epoch: %d' % (epoch_start+1, n_epoch))
    start = time.time()
    
    for epoch in range(epoch_start+1,n_epoch+1):
        
        len_index_all = range(len(au_dataloader_list)) 
        n_batches = 0
        temp_AU_loss = 0
        #for len_index in len_index_all:
#        train_loss_D, train_loss_G, dataloader_len = train_GAN_Seq2Seq(au_generator, au_descriminator, au_dataloader_list[len_index], optimizer_G, optimizer_D)      
        len_index=3
        train_loss, dataloader_len = train_Seq2Seq(au_generator, au_dataloader_list[len_index] , optimizer_G, criterion_model, 1, 0.5)
        print('train: (%f)' % (train_loss/dataloader_len))
        n_batches += dataloader_len
        temp_AU_loss += train_loss
        
        
        print_loss_total += temp_AU_loss
        plot_loss_total += temp_AU_loss     
                   
       
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / (print_every*n_batches)
            print_loss_total = 0
            print('train: (%d %d%%) %.4f' % (epoch, float(epoch-epoch_start) / (n_epoch-epoch_start) * 100, print_loss_avg))
        
        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / (plot_every*n_batches)
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        
#        n_batches = 0
#        temp_AU_loss = 0
#        for len_index in len_index_all:
#            val_loss, dataloader_len, output_tensor = evaluate_Seq2Seq(au_generator, au_dataloader_list[len_index], criterion_model)
#            print('val: (%f)' % (val_loss/dataloader_len))
#            n_batches += dataloader_len
#            temp_AU_loss += val_loss
#        
#        print_loss_total_val += temp_AU_loss
#        plot_loss_total_val += temp_AU_loss
#        
#        if epoch % print_every == 0:
#            print_loss_avg = print_loss_total_val / (print_every*n_batches)
#            print_loss_total_val = 0
#            print('val: (%d %d%%) %.4f' % (epoch, float(epoch-epoch_start) / (n_epoch-epoch_start) * 100, print_loss_avg))
#        
#        if epoch % plot_every == 0:
#            plot_loss_avg = plot_loss_total_val / (plot_every*n_batches)
#            plot_losses_val.append(plot_loss_avg)
#            plot_loss_total_val = 0
        
        print('%s' % (util_func.timeSince(start, float(epoch-epoch_start) / (n_epoch-epoch_start))))        
        
        if epoch % save_every == 0:
            torch.save({
            'epoch': epoch,
            'model_state_dict': au_generator.state_dict(),
            'model_optimizer_state_dict': optimizer_G.state_dict(),
            'train_loss': plot_losses,            
            'val_loss': plot_losses_val,
            }, './checkpoints/Seq2Seq_batch_AU_beta0.8_encdec_LSTM_2_epoch_%d.pth' % epoch)
    
    
    #%%
if model_selection == 'Word2Vec':
    
    SOS_token = np.zeros([mini_batch_size,NUM_AUs])
    SOS_token = torch.as_tensor([[SOS_token]], dtype = torch.float32, device=device).view(1,mini_batch_size,-1)
    #EOS_token = np.ones([mini_batch_size,NUM_AUs])
    #EOS_token = torch.as_tensor([[EOS_token]], dtype = torch.float32, device=device).view(1,mini_batch_size,-1)
    EOS_threshold = 0.05
    
    def train_word2vec_seq(sample, encoder, decoder, EOS_decoder, encoder_optimizer, decoder_optimizer, EOS_decoder_optimizer,
              criterion_AU, criterion_EOS, teacher_forcing_ratio = 0.5, max_length=MAX_LENGTH):
    
        input_tensor = sample['word'].to(device)
        target_tensor = sample['AU_sequence'].to(device)
        len_indicator = sample['len_indicator'].to(device)
        p_id = sample['p_id'].to(device)
        phoneme = sample['phoneme'].to(device)
        
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss_AU = 0
        loss_EOS = 0
        target_length = target_tensor.size(1)
     
        
    #    encoder_hidden = encoder.initHidden(mini_batch_size)
    #    [encoder_outputs, encoder_hidden] = encoder(phoneme,encoder_hidden, mini_batch_size) #.view(1,mini_batch_size,-1).detach()
        
        encoder_hidden = encoder(input_tensor).view(1,mini_batch_size,-1).detach()
        word_pid = encoder_hidden
    #    p_id = p_id.view(1,mini_batch_size,-1)
    #    word_pid = torch.cat((word_embedding, p_id),2).to(device)
    
        decoder_input = SOS_token
        decoder_hidden = decoder.initHidden(mini_batch_size);
        
        EOS_output = torch.zeros(mini_batch_size,1,device=device)
    
    #    EOS_sequence = torch.zeros(mini_batch_size,target_length,1,device=device)
        
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False   
       
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):      
      
            decoder_output, decoder_hidden = decoder(decoder_input, word_pid, decoder_hidden, mini_batch_size)
            loss_AU += criterion_AU.loss_fn(decoder_output, target_tensor[:,di,])        
                    
            if use_teacher_forcing:            
                decoder_input = target_tensor[:,di,].view(1,mini_batch_size,-1)  # Teacher forcing
    
            else:
                decoder_input = decoder_output.detach().view(1,mini_batch_size,-1) 
    
            if(type(decoder_hidden) == tuple):
                EOS_hidden = decoder_hidden[0][1].detach().view(mini_batch_size,-1)   
            else:
                EOS_hidden = decoder_hidden[1].detach().view(mini_batch_size,-1)   
                    
            EOS_output = EOS_output.detach().view(mini_batch_size,-1)
            
    #        current_time = torch.ones(mini_batch_size,1,device=device)*(di+1)
            
            EOS_input = torch.cat((EOS_hidden, EOS_output), 1)
            EOS_output = EOS_decoder(EOS_input)
    #        EOS_sequence[:,di,] = EOS_output
            
            loss_EOS += criterion_EOS(EOS_output, len_indicator[:,di].view(mini_batch_size,-1))
            
            
                     
        loss_AU.backward()
        
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5)
            
        
        loss_EOS.backward()
        
        torch.nn.utils.clip_grad_norm_(EOS_decoder.parameters(), 100)
    
    #    encoder_optimizer.step()
        decoder_optimizer.step()
        EOS_decoder_optimizer.step()
    
    
        return loss_AU.item()/(di+1), loss_EOS.item()/(di+1)
    
    

    n_epoch = 10
    epoch_start = 0
    load_epoch = False
    
    print_every=1
    plot_every=1
    save_every = 1
    
    learning_rate=0.001
    
    hidden_dim = 256
    phoneme_size = 85
    
    vocab_size = len(w2i)
    
    encoder = model_Word2Vec_RNN.word2vec_Encoder(vocab_size, hidden_dim)
    
    #encoder = RNN_Encoder(phoneme_size, hidden_dim, hidden_dim, 'LSTM', num_layers=2)
    
    decoder = model_Word2Vec_RNN.word2vec_Decoder(hidden_dim, NUM_AUs, 'LSTM', device, num_layers=2)
    
    EOS_decoder = model_Word2Vec_RNN.EOS_net(hidden_dim, 1, mini_batch_size)
    
    print ('N parameters: %d' % util_func.count_parameters(decoder))
    
    encoder.to(device)
    decoder.to(device)
    EOS_decoder.to(device)
    
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every 
    
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    
    EOS_decoder_optimizer = optim.Adam(EOS_decoder.parameters(), lr=learning_rate)
    
    
    teacher_forcing_ratio = 0.5
    
    checkpoint = torch.load('./checkpoints/word_phone_vector_999.pth',map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    
    if(load_epoch):
        print('test')
        checkpoint = torch.load('./checkpoints/test218_batch_epoch_1.pth',map_location=device)
        plot_losses = checkpoint['loss']
        epoch_start = checkpoint['epoch']
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
       
    
    #if torch.cuda.device_count() > 1:
    #  print("Let's use", torch.cuda.device_count(), "GPUs!")
    #  encoder = nn.DataParallel(encoder)
    #  decoder = nn.DataParallel(decoder)
    
    criterion_AU = loss_func.loss_cosine_MSE(beta=0.5)
    criterion_EOS = nn.BCELoss()
    
 #%% 
    encoder.train()
    decoder.train()
    
    print('epoch_start: %d n_epoch: %d' % (epoch_start+1, n_epoch))
    start = time.time()
    
    for epoch in range(epoch_start+1,n_epoch+1):
    
        len_index_all = range(len(au_dataloader_list)) 
    #    len_index_all.reverse()
    #    len_index_all = np.random.permutation(len_index_all)
    
        n_batches = 0
    #    for len_index in len_index_all:
        len_index=1
        temp_AU_loss = 0
        temp_EOS_loss = 0
        for i, sample in enumerate(au_dataloader_list[len_index], 0):  
        
            [AU_loss, EOS_loss] = train_word2vec_seq(sample, encoder, decoder, EOS_decoder, encoder_optimizer, decoder_optimizer, EOS_decoder_optimizer, 
                                        criterion_AU, criterion_EOS, teacher_forcing_ratio)
            temp_AU_loss += AU_loss
            temp_EOS_loss += EOS_loss
        print('(%d %d %s) AU_loss %.4f EOS_loss %.4f' % (epoch, i, file_names[len_index][-6:], temp_AU_loss/(i+1), temp_EOS_loss/(i+1)))
    
    
        
        n_batches += i+1   
        print_loss_total += temp_AU_loss
        plot_loss_total += temp_AU_loss 
        
            
        print('%s' % (util_func.timeSince(start, float(epoch-epoch_start) / (n_epoch-epoch_start))))        
        
    #    teacher_forcing_ratio -= 0.01
        
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / (print_every*n_batches)
            print_loss_total = 0
            print(' (%d %d%%) %.4f' % (epoch, float(epoch-epoch_start) / (n_epoch-epoch_start) * 100, print_loss_avg))
        
        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / (plot_every*n_batches)
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        
#        if epoch % save_every == 0:
#            torch.save({
#            'epoch': epoch,
#            'encoder_state_dict': encoder.state_dict(),
#            'decoder_state_dict': decoder.state_dict(),
#            'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
#            'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
#            'loss': plot_losses,            
#            }, './checkpoints/test218_batch_epoch_%d.pth' % epoch)
    
    #%% Elvaluation
                
    SOS_token_test = np.zeros([mini_batch_size,NUM_AUs])
    SOS_token_test = torch.as_tensor([[SOS_token_test]], dtype = torch.float32, device=device).view(1,mini_batch_size,-1)
    #EOS_token_test = np.ones([1,NUM_AUs])
    #EOS_tensor_test = torch.as_tensor([[EOS_token_test]], dtype = torch.float32, device=device).view(1,1,-1)
    EOS_threshold = 0.05
    
    def evaluate(encoder, decoder, EOS_decoder, input_tensor, target_tensor, p_id):
        
        encoder.eval()
        decoder.eval()
        EOS_decoder.eval()
        
        with torch.no_grad():
    
    #        target_length = target_tensor.size(2)
     
            word_embedding = encoder(input_tensor).view(1,mini_batch_size,-1)
            p_id = p_id.view(1,mini_batch_size,-1)
            word_pid = torch.cat((word_embedding, p_id),2).to(device)
    
    
            decoder_input = SOS_token_test
            decoder_hidden = np.zeros([mini_batch_size,hidden_dim])
            decoder_hidden = torch.as_tensor([[decoder_hidden]], dtype = torch.float32, device=device).view(1,mini_batch_size,-1)
    
    
            EOS_output = torch.zeros(mini_batch_size,1,device=device)
            
            EOS_sequence = torch.zeros(MAX_LENGTH,mini_batch_size,1,device=device)
            decoded_AUs = torch.zeros(MAX_LENGTH,mini_batch_size,NUM_AUs,device=device)
    
            for di in range(MAX_LENGTH):
               decoder_output, decoder_hidden = decoder(decoder_input, word_pid, decoder_hidden)
               decoded_AUs[di,] = decoder_output
    #            topv, topi = decoder_output.topk(1)
    #            decoder_input = topi.squeeze().detach()  # detach from history as input
               decoder_input = decoder_output.detach().view(1,mini_batch_size,-1) 
               
               EOS_hidden = decoder_hidden.detach().view(mini_batch_size,-1)   
               EOS_output = EOS_output.detach().view(mini_batch_size,-1)
                
    #           current_time = torch.ones(1,1,device=device)*(di+1)
               
               EOS_input = torch.cat((EOS_hidden, EOS_output), 1)
               EOS_output = EOS_decoder(EOS_input)
               EOS_sequence[di,] = EOS_output
       
    #           if EOS_output > 0.5:
    #              break          
            
            decoded_len = di+1;
            
        return decoded_AUs, decoded_len, EOS_sequence
    
    def evaluateRandomly(encoder, decoder, n=10): 
        sample = next(iter(au_dataloader_list[random.randint(0,27)]))
        print(sample['word'][0,].item(), i2w[sample['word'][0,].item()])
        
        input_tensor = sample['word'].to(device)
        target_AUs = sample['AU_sequence'].to(device)
        p_id = sample['p_id'].to(device)
        
    #    target_EOS = sample['len_indicator'][0,].to(device)
        
        [decoded_AUs, decoded_len, EOS_sequence] = evaluate(encoder, decoder, EOS_decoder, input_tensor, target_AUs, p_id)
        print(decoded_len,target_AUs.size()[0])
    
#%%    
#m = nn.Sigmoid()
#loss = nn.BCELoss()
#input = torch.ones(1)*0.9
#target = torch.zeros(1)
#output = loss(input, target)
#output.backward()

#%%
#class word_embedding(Dataset):
#    def __init__(self):
#        super(word_embedding,self).__init__()
#        self._dataset_size = transformed.shape[0]
#
#    def __len__(self):
#        return self._dataset_size
#    
#    def __getitem__(self,index):
#        assert (index < self._dataset_size)
#        
#        word = torch.as_tensor(index, dtype = torch.long)
#        phone_vector = transformed[index,:]
#        phone_vector = torch.as_tensor(phone_vector, dtype = torch.float32)
#        
#        sample = {  'word': word,
#                    'phone_vector': phone_vector}
#        
#        return sample
#
#word_dataset = word_embedding()
#word_dataloader = DataLoader(word_dataset, batch_size=30, shuffle=True, drop_last=True)
#
#for i, sample in enumerate(word_dataloader):
#    print(sample)
#    if i==0:
#        break
    
#%% Pre-Train word to phone
#word_criterion = torch.nn.MSELoss(size_average=None)
#word_encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
#
#
#for epoch in range(1,1000):
#    total_loss = 0;
#    for i, sample in enumerate(word_dataloader):
#        
#        word_encoder_optimizer.zero_grad()
#        
#        input_tensor = sample['word'].to(device)
#        target_tensor = sample['phone_vector'].to(device)
#        
#        word_embedding = encoder(input_tensor)
#            
#        loss = word_criterion(word_embedding,target_tensor)
#        total_loss += loss
#        
#        loss.backward()
#        word_encoder_optimizer.step()
#        
#    print('%d: %f' % (epoch,total_loss))
#
#torch.save({
#        'epoch': epoch,
#        'encoder_state_dict': encoder.state_dict(),        
#        'encoder_optimizer_state_dict': word_encoder_optimizer.state_dict(),    
#        'loss': total_loss,            
#        }, './checkpoints/word_phone_vector_%d.pth' % epoch)
