# -*- coding: utf-8 -*-

import numpy as np
import torch

import scipy.io as sio

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from torch.utils.data import Dataset

MAX_LENGTH = 40
EOS_IDX = 40
PAD_IDX = 0

class word_AU_Dataset(Dataset):
    def __init__(self, opt, file_name, w2i, word_to_symbol):
        super(word_AU_Dataset, self).__init__()
        self._opt = opt
        self._file_names = file_name
        self._word_AUs = sio.loadmat(file_name)
        self._dataset_size = len(self._word_AUs['t_data_train'])
        self._w2i = w2i
        self._word_to_symbol = word_to_symbol
        
#        self._train_word = mode_AUs['train_word']
#        self._mode_AUs = mode_AUs['mode_AUs']
#        self._dataset_size = len(mode_AUs['train_word'])
    
    def __len__(self):
        return  self._dataset_size 
        
    def __getitem__(self, index):
        assert (index < self._dataset_size)        
       
        word_gt = self._w2i[self._word_AUs['t_data_train'][index,0][0]]
#        word_gt.append(84)       
       
        
        word_gt = torch.as_tensor(word_gt, dtype = torch.long)
        
        phoneme_gt = list(self._word_to_symbol[self._word_AUs['t_data_train'][index,0][0]]) 
        phoneme_gt = [item+1 for item in phoneme_gt]
        phoneme_gt.append(EOS_IDX)
        
        phoneme_len = len(phoneme_gt)
        phoneme_gt.extend([PAD_IDX]*(MAX_LENGTH - phoneme_len))
        phoneme_gt = torch.as_tensor(phoneme_gt, dtype = torch.long)
        
        t_data = self._word_AUs['t_data_train'][index,1]
         
#        temp = temp[random.randint(0, len(temp) - 1)]

        temp_AUs = t_data[:,5:21]/5
        temp_AUs = temp_AUs.tolist()
#        temp = np.vstack([temp,np.ones([1,NUM_AUs])])
        
        AU_sequence_gt = torch.as_tensor(temp_AUs, dtype = torch.float32)
        
        temp_pose = t_data[:,25:28]
        temp_pose = temp_pose.tolist()
        pose_sequence_gt = torch.as_tensor(temp_pose, dtype = torch.float32)

        
        seq_len = AU_sequence_gt.size(0)
        len_indicator = np.zeros(seq_len)
        len_indicator[-1] = 1
        
        len_indicator = torch.as_tensor(len_indicator, dtype = torch.float32)
        
        p_id = self._word_AUs['t_data_train'][index,3]
        p_id = torch.as_tensor(p_id, dtype = torch.float32)
        
        
        sample = {
                    'word': word_gt,
                    'phoneme': phoneme_gt,
                    'phoneme_len': phoneme_len,
                    'AU_sequence': AU_sequence_gt,
                    'pose_sequence': pose_sequence_gt,
                    'len_indicator': len_indicator,
                    'p_id': p_id
                 }
        
        return sample



class word_AU_Dataset_val(Dataset):
    def __init__(self, opt, file_name, w2i, word_to_symbol):
        super(word_AU_Dataset_val, self).__init__()
        self._opt = opt
        self._file_names = file_name
        self._word_AUs = sio.loadmat(file_name)
        self._dataset_size = len(self._word_AUs['t_data_val'])
        self._w2i = w2i
        self._word_to_symbol = word_to_symbol
        
#        self._train_word = mode_AUs['train_word']
#        self._mode_AUs = mode_AUs['mode_AUs']
#        self._dataset_size = len(mode_AUs['train_word'])
    
    def __len__(self):
        return  self._dataset_size 
        
    def __getitem__(self, index):
        assert (index < self._dataset_size)        
       
        word_gt = self._w2i[self._word_AUs['t_data_val'][index,0][0]]
#        word_gt.append(84)       
       
        
        word_gt = torch.as_tensor(word_gt, dtype = torch.long)
        
        phoneme_gt = list(self._word_to_symbol[self._word_AUs['t_data_val'][index,0][0]]) 
        phoneme_gt = [item+1 for item in phoneme_gt]
        phoneme_gt.append(EOS_IDX)
        
        phoneme_len = len(phoneme_gt)
        phoneme_gt.extend([PAD_IDX]*(MAX_LENGTH - phoneme_len))
        phoneme_gt = torch.as_tensor(phoneme_gt, dtype = torch.long)
        
        temp = self._word_AUs['t_data_val'][index,1]/5
         
#        temp = temp[random.randint(0, len(temp) - 1)]

        temp = temp[:,5:21]
        temp = temp.tolist()
#        temp = np.vstack([temp,np.ones([1,NUM_AUs])])
        
        AU_sequence_gt = torch.as_tensor(temp, dtype = torch.float32)
        
        seq_len = AU_sequence_gt.size(0)
        len_indicator = np.zeros(seq_len)
        len_indicator[-1] = 1
        
        len_indicator = torch.as_tensor(len_indicator, dtype = torch.float32)
        
        p_id = self._word_AUs['t_data_val'][index,3]
        p_id = torch.as_tensor(float(p_id), dtype = torch.long)
        
        
        sample = {
                    'word': word_gt,
                    'phoneme': phoneme_gt,
                    'phoneme_len': phoneme_len,
                    'AU_sequence': AU_sequence_gt,
                    'len_indicator': len_indicator,
                    'p_id': p_id
                 }
        
        return sample

