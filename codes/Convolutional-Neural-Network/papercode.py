# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 17:45:15 2016

@author: rehan
"""
## Clear function
__saved_context__ = {}
def saveContext():
    import sys
    __saved_context__.update(sys.modules[__name__].__dict__)

def restoreContext():
    import sys
    names = sys.modules[__name__].__dict__.keys()
    for n in names:
        if n not in __saved_context__:
            del sys.modules[__name__].__dict__[n]

clear = restoreContext
saveContext()
clear()  # Calling clear for clearing workspace

import numpy as np
import time
import matplotlib.pyplot as plt
import pylab
import wave
import struct
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from scipy.signal import decimate
import copy
import matplotlib
import stft
import scipy
import sklearn

def windowing_signal(data,window_len,stride):
    wstart =0
    wend = window_len
    stride = stride
    nof = data.shape[1]
    wd = []
    while wend <= nof:
        dtemp = data[:,wstart:wend]
        wd.append(dtemp)
        wstart = wstart + stride
        wend = wend + stride
    return wd

def readwave(filepath):
    f1 = wave.open(filepath,'r')
    nof  = wave.Wave_read.getnframes(f1)
    frate = wave.Wave_read.getframerate(f1)
    nbytes = wave.Wave_read.getsampwidth(f1)
    x=np.zeros((nof,))
    for i in range(nof):
        s1 = wave.Wave_read.readframes(f1,1)
        us1 = struct.unpack("h",s1) #h: 16bit sample each
        x[i] = us1[0]
    f1.close()
    return x,frate,nof,nbytes

def writewave(path,data,samp_freq,samp_width,nchannels):
    fw = wave.open(path,'w')
    params=[nchannels,samp_width,samp_freq,len(data),'NONE','not compressed']
    wave.Wave_write.setparams(fw,params)
    for i in range(len(data)):
        wave.Wave_write.writeframes(fw,struct.pack('h',data[i]))
    fw.close()

xf,f1rate,nof1,xfbytes = readwave('./AudioDataFiles/fsew0_001-043.wav')
xm,f2rate,nof2,xmbytes = readwave('./AudioDataFiles/msak0_001-043.wav')
#xfnor = sklearn.preprocessing.scale(xf)
#xmnor = sklearn.preprocessing.scale(xm)
newfrate = 4000
xf_deci = decimate(xf,f1rate/newfrate)
xm_deci = decimate(xm,f2rate/newfrate)

xfnor = xf_deci/2.0**15 # normalizing...
xmnor = xm_deci/2.0**15 

xfc_train = copy.copy(xfnor[0:newfrate*120]) # cropping for training data...
xmc_train = copy.copy(xmnor[0:newfrate*120]) 

xf_test = xfnor[newfrate*120:xfnor.size] # cropping for testing data...
xm_test = xmnor[newfrate*120:xfnor.size] 

mix_test = xf_test + xm_test
mix_train = xfc_train+xmc_train

######## Writing a wavefile ############
#writewave('./male_test.wav',xm_test,f1rate,2,1)
########################################
#Fspec = pylab.specgram(xfc_train,NFFT=128,Fs=f1rate,noverlap=64,mode='magnitude',\
#scale='dB')
#Mspec = pylab.specgram(xmc_train,NFFT=128,Fs=f2rate,noverlap=64,mode='magnitude',\
#scale='dB')
#mixspec_train = pylab.specgram(mix_train,NFFT=128,Fs=f2rate,noverlap=64,mode='magnitude',\
#scale='dB')
#Female_test = pylab.specgram(xf_test,NFFT=128,Fs=f1rate,noverlap=64,\
#mode='magnitude',scale='dB')
#Male_test = pylab.specgram(xm_test,NFFT=128,Fs=f2rate,noverlap=64,\
#mode='magnitude',scale='dB')
########################################

Fspec = stft.spectrogram(xfc_train,framelength=128,hopsize=16,\
window=scipy.signal.hanning)
Fspec = np.absolute(Fspec)

Mspec = stft.spectrogram(xmc_train,framelength=128,hopsize=16,\
window=scipy.signal.hanning)
Mspec = np.absolute(Mspec)

mixspec_train = stft.spectrogram(mix_train,framelength=128,hopsize=16,\
window=scipy.signal.hanning)
mixspec_train = np.absolute(mixspec_train)

mixspec_test = stft.spectrogram(mix_test,framelength=128,hopsize=16,\
window=scipy.signal.hanning)
mixspec_test = np.absolute(mixspec_test)

binary_mask = np.array(Mspec > Fspec)#,dtype=int)
mixspec_train_norm = (mixspec_train-mixspec_train.min())/(mixspec_train.max()\
-mixspec_train.min())
mixspec_test_norm = (mixspec_test-mixspec_test.min())/(mixspec_test.max()\
-mixspec_test.min())

##Windowing the mixture spectrogram & binary Mask##
mixspec_train_wind = windowing_signal(mixspec_train_norm,20,10)
binarydata = windowing_signal(binary_mask,20,10)
mixspec_test_wind = windowing_signal(mixspec_test_norm,20,1)

#### Reshaping the training data and labels for CNN ###### 
training_data = np.zeros((len(mixspec_train_wind),1,65,20))
t_labels = np.zeros((len(mixspec_train_wind),65,20))
for i in range(len(mixspec_train_wind)):
    training_data[i,0,:,:] = mixspec_train_wind[i]
    t_labels[i,:,:] = binarydata[i] 
training_labels = np.reshape(t_labels,(-1,65*20))
#################################################
#### Reshaping the test data ###### 
test_data = np.zeros((len(mixspec_test_wind),1,65,20))
for i in range(len(mixspec_test_wind)):
    test_data[i,0,:,:] = mixspec_test_wind[i]

net1 = NeuralNet(layers=[('input',layers.InputLayer),\
                        ('conv2d1',layers.Conv2DLayer),\
                        ('maxpool1',layers.MaxPool2DLayer),\
#                        ('conv2d2',layers.Conv2DLayer),\
#                        ('maxpool2',layers.MaxPool2DLayer),\
                        ('hidden',layers.DenseLayer),     
                        ('output',layers.DenseLayer)],\
                        # Input layer...                        
                        input_shape = (None,1,65,20),\
                        # layer conv2d1...
                        conv2d1_num_filters=1,\
                        conv2d1_filter_size=(3,3),\
                        # layer maxpool1
                        maxpool1_pool_size=(2,2),\
                        # layer conv2d2
#                        conv2d2_num_filters =1,\
#                        conv2d2_filter_size = (3,3),\
                        #layer maxpool2
#                        maxpool2_pool_size=(2,2),\
                        # hidden layer
                        hidden_num_units = 1300,\
                        hidden_nonlinearity = lasagne.nonlinearities.sigmoid,\
                        # output
                        output_num_units = 1300,\
                        output_nonlinearity = lasagne.nonlinearities.sigmoid,\
                        # optimizaiton param methods
                        regression = True,\
                        objective_loss_function=lasagne.objectives.binary_crossentropy,\
                        update = nesterov_momentum,\
                        update_learning_rate = 0.01,\
                        update_momentum = 0.9,\
                        max_epochs = 100,\
                        verbose=1,
                        )

nn=net1.fit(training_data,training_labels)
pred = nn.predict(test_data)
#visualize.plot_loss(nn)

pred_data = np.zeros((pred.shape[0],65,20))
pred_data = np.reshape(pred,(-1,65,20))

add = pred_data[0,:,:]
for i in range(pred_data.shape[0]-1):
#    print i,add.shape,np.concatenate((add,np.zeros((65,1))),axis=1).shape,\
#    np.concatenate(((np.zeros((65,i+1))),test_data[i+1,0,:,:]),axis=1).shape
    add = np.concatenate((add,np.zeros((65,1))),axis=1)\
    + np.concatenate(((np.zeros((65,i+1))),pred_data[i+1,:,:]),axis=1)

avg_out = add/20.0
alpha = 0.5
Male_binary_out = np.array(avg_out > alpha)#,dtype=int)
Female_binary_out = np.array(avg_out < (1-alpha))#,dtype=int)

xf_test = xf_deci[newfrate*120:xfnor.size] # original samples not noramalized.
xm_test = xm_deci[newfrate*120:xfnor.size] 
mix_test = np.short(xf_test + xm_test)
mixspec_test = stft.spectrogram(mix_test,framelength=128,hopsize=16,\
window=scipy.signal.hanning)

Male_output = Male_binary_out*(mixspec_test)
Female_output = Female_binary_out*(mixspec_test)

male_audio_recover = stft.ispectrogram(Male_output,framelength=128,hopsize=16,\
window=scipy.signal.hanning)
female_audio_recover = stft.ispectrogram(Female_output,framelength=128,hopsize=16,\
window=scipy.signal.hanning)

writewave('./male_recovered.wav',male_audio_recover,f1rate,2,1)
writewave('./female_recovered2.wav',np.short(female_audio_recover),f1rate,2,1)

#pylab.pcolormesh(Male_binary_out*(10*np.log10(xmixspectest[:,1:-3])))
#pylab.pcolormesh(np.nan_to_num(10*np.log10(Female_output)))
################################################

