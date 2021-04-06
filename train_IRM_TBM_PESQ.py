# -*- coding: utf-8 -*-

from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Dropout, Flatten, Activation, SpatialDropout2D, Reshape, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers import LSTM, TimeDistributed, Bidirectional, dot, Input, Concatenate, Multiply, Subtract, Maximum
from keras.layers.pooling import GlobalAveragePooling2D
from joblib import Parallel, delayed
from support.SpectralNormalizationKeras import DenseSN, ConvSN1D, ConvSN2D, ConvSN3D

import shutil
import scipy.io
import librosa
import os
import time
import numpy as np
import numpy.matlib
import random
import subprocess

random.seed(999)
Target_score=np.asarray([1.0])
epoch=200
num_sample=2
mask_min = 0.05

def generator(file_list):
    index=0
    while True:
        wav_name = file_list[index]
        noisy_path = 'your path/mask-fusion/train/noisy/'+wav_name+'.wav'
        clean_path = 'your path/mask-fusion/train/clean/'+wav_name+'.Clean.wav'
        noise_path = 'your path/mask-fusion/train/noise/'+wav_name+'.Noise.wav'
        noisy = librosa.load(noisy_path,sr=16000)
        noise = librosa.load(noise_path,sr=16000)
        clean = librosa.load(clean_path,sr=16000)
        noisy_LP_normalization, _, _, _= Sp_and_phase(noisy[0], Normalization=True)
        noisy_LP, _, _, _= Sp_and_phase(noisy[0], Normalization=False)
        clean_LP_normalization, _, _, TBM_threshold= Sp_and_phase(clean[0], Normalization=True)
        clean_LP, _, _, _= Sp_and_phase(clean[0])
        noise_LP, _, _, _= Sp_and_phase(noise[0])
        clean_pow = np.square(clean_LP)
        noise_pow = np.square(noise_LP)
        IRM_target = np.sqrt(clean_pow/(clean_pow+noise_pow))
        #IRM_target = clean_LP/(noise_LP+clean_LP)
        TBM_target = np.float64(clean_LP_normalization>TBM_threshold)
        index += 1
        yield [noisy_LP_normalization, noisy_LP.reshape((1,257,noisy_LP.shape[1],1)), clean_LP.reshape((1,257,noisy_LP.shape[1],1)), mask_min*np.ones((1,257,noisy_LP.shape[1],1))],[IRM_target, TBM_target, Target_score]

def gen_flist(data_dir):
    flist=[]
    for line in  open(data_dir,'r'):
        flist.append(line.split('\n')[0])
    return flist

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


print ('Reading path of training data...')
Generator_Train_Noisy_paths = gen_flist('your path/mask-fusion/file_list/tr.txt')
random.shuffle(Generator_Train_Noisy_paths)

print ('Enhancement-net constructuring...')
'''
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
'''
ge_model=load_model('your path/mask-fusion/model/test1.h5')
ge_model.summary()

print ('PESQ-net constructuring...')
'''
_input = Input(shape=(257,None,2))
_inputBN = BatchNormalization(axis=-1)(_input)

C1=ConvSN2D(15, (5,5), padding='valid',  data_format='channels_last') (_inputBN)
C1=LeakyReLU()(C1)

C2=ConvSN2D(25, (7,7), padding='valid',  data_format='channels_last') (C1)
C2=LeakyReLU()(C2)

C3=ConvSN2D(40, (9,9), padding='valid',  data_format='channels_last') (C2)
C3=LeakyReLU()(C3)

C4=ConvSN2D(50, (11,11), padding='valid',  data_format='channels_last') (C3)
C4=LeakyReLU()(C4)

Average_score=GlobalAveragePooling2D(name='Average_score')(C4)  #(batch_size, channels)

D1=DenseSN(50)(Average_score)
D1=LeakyReLU()(D1)

D2=DenseSN(10)(D1)
D2=LeakyReLU()(D2)

Score=DenseSN(1)(D2)

Discriminator = Model(outputs=Score, inputs=_input)
'''
Discriminator = load_model('your path/mask-fusion/model/PESQ_net.h5',custom_objects={'ConvSN2D':ConvSN2D,'DenseSN':DenseSN})
Discriminator.summary()

Discriminator.trainable = False
Clean_reference = Input(shape=(257,None,1),name='input_2')
Noisy_LP        = Input(shape=(257,None,1),name='input_3')
Min_mask        = Input(shape=(257,None,1),name='input_4')

[IRM_output,TBM_output] = ge_model.output
Reshape_ge_model_output=Reshape((257, -1, 1))(IRM_output)
Mask=Maximum()([Reshape_ge_model_output, Min_mask])

Enhanced = Multiply()([Mask, Noisy_LP])
Discriminator_input= Concatenate(axis=-1)([Enhanced, Clean_reference]) # Here the input of Discriminator is (Noisy, Clean) pair, so a clean reference is needed!!
Predicted_score=Discriminator(Discriminator_input) 

mtl_model= Model(inputs=[ge_model.input, Noisy_LP, Clean_reference, Min_mask], outputs=[IRM_output,TBM_output,Predicted_score])
mtl_model.compile(loss={'activation_1':'mse','activation_2':'binary_crossentropy','model_1':'mse'}, loss_weights={'activation_1':1,'activation_2':0.1,'model_1':10}, optimizer='adam')
mtl_model.summary()
######## Model define end #########

for current_epoch in np.arange(1, epoch+1):
    random.shuffle(Generator_Train_Noisy_paths)
    g = generator(Generator_Train_Noisy_paths[0:num_sample])
    mtl_model.fit_generator(g, steps_per_epoch=num_sample, epochs=1, verbose=1, max_queue_size=1, workers=1)
    ge_model.save('your path/mask-fusion/model/test2.h5')
