#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 09:53:42 2019

@author: zheng.1443
"""

#%%
file_name = './phone_data/cmudict-0.7b.symbols.txt'
f_handler = open(file_name,'r')

phone_symbol = dict()

i=0
for line in f_handler:
    line = line.strip()
    phone_symbol[line] = i
    i = i+1

f_handler.close()

#%%

file_name = './phone_data/cmudict-0.7b.txt'
f_handler = open(file_name,'rb',encoding='latin1')

word_cmu=[]
phone_cmu=[]
symbols_cmu=[]

word_to_phone = dict()
word_to_symbol = dict()

for line in f_handler: #enumerate(sys.stdin):
    if line.startswith(';'):
        continue
    line = line.strip()
    word, phones = line.split("  ")
    
    phones = phones.split()
    symbols = [phone_symbol[index] for index in phones]
    
    word_cmu.append(word)
    phone_cmu.append(phones)
    symbols_cmu.append(symbols)
    word_to_phone[word]=phones
    word_to_symbol[word]=symbols

f_handler.close()

#%%
import pickle

f_handler = open('word_to_phone_symbol.pkl','w')
pickle.dump([word_to_phone, word_to_symbol],f_handler)
f_handler.close()


