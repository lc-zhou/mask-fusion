# -*- coding: utf-8 -*-

from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Dropout, Activation, Reshape, Lambda
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from joblib import Parallel, delayed
from keras import backend as K
from keras.layers import LSTM, Conv1D, TimeDistributed, Bidirectional, dot, Input, Concatenate, Multiply, Subtract, Maximum

import librosa
import os
import numpy as np
import random
import scipy.io
import subprocess

def generator(file_flist):
    index=0
    while True:
        noisy_path = file_flist[index]
        file_name = noisy_path.split('/')[-1]
        env_name = noisy_path.split('/')[-2]
        wave_name = file_name.split('.')[0]
        clean_path = '/workspace/CHiME3/data/audio/16kHz/isolated_ext/'+env_name+'/'+wave_name+'.CH1.Clean.wav'
        noise_path = '/workspace/CHiME3/data/audio/16kHz/isolated_ext/'+env_name+'/'+wave_name+'.CH1.Noise.wav'
        noisy = librosa.load(noisy_path,sr=16000)
        noise = librosa.load(noise_path,sr=16000)
        clean = librosa.load(clean_path,sr=16000)
        noisy_LP_normalization, _, _, _= Sp_and_phase(noisy[0], Normalization=True)
        clean_LP_normalization, _, _, TBM_threshold= Sp_and_phase(clean[0], Normalization=True)
        clean_LP, _, _, _= Sp_and_phase(clean[0])
        noise_LP, _, _, _= Sp_and_phase(noise[0])
        clean_pow = np.square(clean_LP)
        noise_pow = np.square(noise_LP)
        IRM_target = np.sqrt(clean_pow/(clean_pow+noise_pow))
        #IRM_target = clean_LP/(noise_LP+clean_LP)
        TBM_target = np.float64(clean_LP_normalization>TBM_threshold)
        index += 1
        yield noisy_LP_normalization, [IRM_target,TBM_target]

def Sp_and_phase(signal, Normalization=False):        
    signal_length = signal.shape[0]
    n_fft = 512
    y_pad = librosa.util.fix_length(signal, signal_length + n_fft // 2)
    F = librosa.stft(y_pad, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)
    Lp=np.abs(F)
    phase=np.angle(F)
    if Normalization==True:    
        meanR = np.mean(Lp, axis=1).reshape((257,1))
        stdR = np.std(Lp, axis=1).reshape((257,1))+1e-12
        NLp = (Lp-meanR)/stdR
        threshold = 0
        #threshold = np.reshape(-0.6*meanR/stdR,(1,1,257))#-0.6 -> 20%
    else:
        threshold = 0
        NLp=Lp
    NLp=np.reshape(NLp.T,(1,NLp.shape[1],257))
    return NLp, phase, signal_length, threshold

def gen_flist(data_dir):
    flist=[]
    for line in open(data_dir,'r'):
        path=line.split('\n')[0]
        flist.append(path)
    return flist

random.seed(999)
num_sample=200
epoch=200
mask_min=0.05

data = Input(shape=(None, 257))

d1 = Bidirectional(LSTM(200, return_sequences=True), merge_mode='concat')(data)
d2 = Bidirectional(LSTM(200, return_sequences=True), merge_mode='concat')(d1)

n1 = TimeDistributed(Dense(300))(d2)
n2 = LeakyReLU()(n1)
n3 = Dropout(0.05)(n2)

t1 = TimeDistributed(Dense(300))(n3)
t2 = LeakyReLU()(t1)
t3 = Dropout(0.05)(t2)

x1 = TimeDistributed(Dense(257))(t3)
IRM_mask = Activation('sigmoid')(x1)

x2 = TimeDistributed(Dense(257))(t3)
TBM_mask = Activation('sigmoid')(x2)

ge_model = Model(inputs=data, outputs=[IRM_mask, TBM_mask])
ge_model.compile(loss={'activation_1':'mse','activation_2':'binary_crossentropy'}, loss_weights={'activation_1':1,'activation_2':0.1}, optimizer='adam')
ge_model.summary()

print('load data for training')
Generator_Train_paths = gen_flist('/home/zlc/masks-fusion/data_txt/tr.txt')

print('training')
for epoch_g in np.arange(1,epoch+1):
    print('current epoch '+str(epoch_g))
    random.shuffle(Generator_Train_paths)
    g = generator(Generator_Train_paths)
    ge_model.fit_generator(g, steps_per_epoch=num_sample, epochs=1, verbose=1, max_queue_size=1, workers=1)
    ge_model.save('/home/zlc/masks-fusion/model/test1.h5')





