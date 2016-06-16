# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 01:41:55 2016

@author: rehan

ANN MNIST
specifications: 
input nodes = 784
hidden layers = 2
hidden layer nodes = 50
output nodes = 1
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import cPickle, gzip

print("Loading data set...")
try:
    train_set and test_set and valid_set
except NameError:
    f = gzip.open('./mnist.pkl.gz','rb')
    train_set, valid_set, test_set =cPickle.load(f)
    f.close()
else:
    print("Data Already loaded in Variable explorer...")

tr_label = np.concatenate([train_set[1][train_set[1]==0],train_set[1][train_set[1]==1]])
tr_data = np.concatenate([train_set[0][train_set[1]==0],train_set[0][train_set[1]==1]])
tst_label = np.concatenate([test_set[1][test_set[1]==0],test_set[1][test_set[1]==1]])
tst_data = np.concatenate([test_set[0][test_set[1]==0],test_set[0][test_set[1]==1]])

theta1 = 0.01*np.random.random_sample((50,785))
theta2 = 0.01*np.random.random_sample((50,51))
theta3 = 0.01*np.random.random_sample((1,51))

def acfunc(z):
    return 1.0/(1.0+np.exp(-z))

def forward_pass(bias,ins,t1,t2,t3):
    br = np.repeat(bias,len(ins))
    x=np.concatenate((np.reshape(br,(len(ins),1)),ins),axis=1)
    a2 = acfunc(np.dot(t1,x.T))
    a2 = np.concatenate((np.reshape(np.repeat(bias,a2.shape[1]),(1,a2.shape[1])),a2))
    a3 = acfunc(np.dot(t2,a2))
    a3 = np.concatenate((np.reshape(np.repeat(bias,a3.shape[1]),(1,a3.shape[1])),a3))
    aout = acfunc(np.dot(t3,a3))
    return a2,a3,aout

def back_prop(outs,a2,a3,est_out,t1,t2,t3):
    delta4 = est_out.T-np.reshape(outs,(len(outs),1))
    delta3 = (t3[0,1:51]*delta4).T*a3[1:51,:]*(1-a3[1:51,:])
    delta2 = np.dot(t2[:,1:51].T,delta3)*a2[1:51,:]*(1-a2[1:51,:]) 
    return delta4,delta3,delta2

def cal_D(bias,alpha,ins,a2,a3,delta2,delta3,delta4,m,t1,t2,t3,lemda):
    br = np.repeat(bias,len(ins))
    x=np.concatenate((np.reshape(br,(len(ins),1)),ins),axis=1)
    t1 = t1 - (1.0/m)*alpha*(np.dot(delta2,x))      - (lemda/m)*t1
    t2 = t2 - (1.0/m)*alpha*(np.dot(delta3,a2.T))   - (lemda/m)*t2
    t3 = t3 - (1.0/m)*alpha*(np.dot(delta4.T,a3.T)) - (lemda/m)*t3
    return t1,t2,t3

start_time = time.time()
alpha = 0.5
bias = np.array([0.5])
no_of_iter = 200
lemda = 0.1
m = len(tr_data)
error = np.zeros((m,1))

print "number of nodes=",[784, 50, 50, 1]
print "bias  = ",bias
print "alpha = ",alpha
print "Lembda = ",lemda

error_train = np.zeros((no_of_iter,1))

for it in range(no_of_iter):

    a2,a3,est_out = forward_pass(bias,tr_data,theta1,theta2,theta3)    
    d4,d3,d2 = back_prop(tr_label,a2,a3,est_out,theta1,theta2,theta3)
    theta1,theta2,theta3 = cal_D(bias,alpha,tr_data,a2,a3,d2,d3,d4,m,theta1,theta2,theta3,lemda)
    a2,a3,est = forward_pass(bias,tr_data,theta1,theta2,theta3)
    error_train[it] = (1.0/m)*np.sum(np.square(est-tr_label))

#a2,a3,est_out = forward_pass(bias,tr_data,theta1,theta2,theta3)
#error_train = np.square(est_out-tr_label)
#print "Training accuracy is = ", 100-(1.0/m)*np.sum(error_train)

#a2,a3,est_out = forward_pass(bias,tst_data,theta1,theta2,theta3)
#error_test = np.square(est_out-tst_label)
#print "Testing accuracy is = ", 100-(1.0/len(tst_label))*np.sum(error_train)

print "Training accuracy is = ", 100-100*error_train[no_of_iter-1]
elapsed_time = time.time() - start_time
plt.plot(error_train)
print "Elapsed simulation time = ",elapsed_time,"sec"
