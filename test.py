# -*- coding: utf-8 -*-

from support.signal_processing import audiowrite
from keras.models import load_model

import librosa
import numpy as np
import scipy

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
    else:
        NLp=Lp
    NLp=np.reshape(NLp.T,(1,NLp.shape[1],257))
    return NLp, phase, signal_length

def SP_to_wav(mag, phase, signal_length):
    Rec = np.multiply(mag , np.exp(1j*phase))
    result = librosa.istft(Rec,
                           hop_length=256,
                           win_length=512,
                           window=scipy.signal.hamming, length=signal_length)
    return result

def update_mask(irm, tbm, thre=0.87, supp=2): #
    frames = irm.shape[1]
    feq_pin = irm.shape[2]
    irm_tbm = np.zeros(irm.shape, dtype=np.float32)
    for i in range(frames):
        for j in range(feq_pin):
            m1 = irm[0,i,j]
            m2 = tbm[0,i,j]
            if m2>=thre:
                irm_tbm[0,i,j]=m1
            else:
                irm_tbm[0,i,j]=m1/supp
    return irm_tbm

def gen_flist(data_dir):
    flist=[]
    for line in open(data_dir,'r'):
        path=line.split('\n')[0]
        flist.append(path)
    return flist

print('load model....')
ge_model = load_model('/home/zlc/masks-fusion/model/SE_MTL1_IRM_TBM.h5')
ge_model.summary()

print('load data for testing')
Generator_Test_paths = gen_flist('/home/zlc/masks-fusion/data_txt/et.txt')
mask_min = 0.05

print('enhance the wave')
for noisy_path in Generator_Test_paths:
    file_name = noisy_path.split('/')[-1]
    env=noisy_path.split('/')[-2]
    wave_name = file_name.split('.')[0]
    sec=env.split('_')[1]
    noisy = librosa.load(noisy_path,sr=16000)
    noisy_LP_normalization, Nphase, signal_length = Sp_and_phase(noisy[0], Normalization=True)
    noisy_LP, _, _= Sp_and_phase(noisy[0])
    [IRM, TBM] = ge_model.predict(noisy_LP_normalization)
    IRM = update_mask(IRM,TBM,0.87,2)
    #IRM = ge_model.predict(noisy_LP_normalization) # for SE_IRM,SE_TBM model
    mask = np.maximum(IRM, mask_min)
    E=np.squeeze(noisy_LP*mask)
    enhanced_wav=SP_to_wav(E.T,Nphase, signal_length)
    enhanced_wav=enhanced_wav/np.max(abs(enhanced_wav))
    output_path = '/home/zlc/masks-fusion/enhanced/'+wave_name+'_enhanced.wav'
    audiowrite(enhanced_wav.astype(np.float32),output_path,16000)