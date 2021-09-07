#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 15:42:01 2019

@author: zheng.1443
"""

import time
import math

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker


#from . import (
#    FairseqEncoder, FairseqIncrementalDecoder, FairseqModel,
#    FairseqLanguageModel, register_model, register_model_architecture,
#)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (~ %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



#%%
#import csv
##
#with open('./word.csv', 'wb') as myfile:
#    wr = csv.writer(myfile, delimiter=",",quoting=csv.QUOTE_ALL)
#    wr.writerow(target_tensor.squeeze().cpu().numpy())

#import numpy
#temp = output_tensor.squeeze().cpu().detach().numpy()
#numpy.savetxt("foo.csv", temp, delimiter=",")

#    
##%%
#import pickle
#
#f_handler = open('word_to_phone_symbol.pkl','r')
#[word_to_phone, word_to_symbol] = pickle.load(f_handler)
#f_handler.close()
#
#
#f_handler = open('word_to_phone_vector.pkl','w')
#pickle.dump([transformed],f_handler)
#f_handler.close()
