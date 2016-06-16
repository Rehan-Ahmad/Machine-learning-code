# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 11:10:20 2016

@author: rehan
Logistic Regression on MNIST Data set. 
"""
import numpy as np
import cPickle, gzip,copy
import time
import matplotlib.pyplot as plt

start_time = time.time()
print("Laoding data set...")

try:
    train_set and test_set and valid_set
except NameError:
    f = gzip.open('./mnist.pkl.gz','rb')
    train_set, valid_set, test_set =cPickle.load(f)
    f.close()
else:
    print("Data Already loaded in Variable explorer...")

train_label01 = np.concatenate([train_set[1][train_set[1]==0],train_set[1][train_set[1]==1]])
train_data01 = np.concatenate([train_set[0][train_set[1]==0],train_set[0][train_set[1]==1]])
test_label01 = np.concatenate([test_set[1][test_set[1]==0],test_set[1][test_set[1]==1]])
test_data01 = np.concatenate([test_set[0][test_set[1]==0],test_set[0][test_set[1]==1]])

theta = 2*np.random.random_sample((785,1))-1

def GD(x,y,t):
    alpha=0.1
    m = 10610.0
    er = np.zeros((100,1))
    x = np.concatenate((np.ones((10610,1)),x,),axis = 1)
    loop = 0

    while loop <= len(er)-1:
        h = np.array(hyp(x,t)).T
        h_y = 0.0   
        for j in range(785):
            h_y = np.sum((h[:,0]-y)*x[:,j])
            t[j,0] = t[j,0] - alpha*h_y*(1.0/m)
        er[loop] = cost(x[:,1::],y,t)
        print "iter=",loop,"error = ",er[loop]
        if er[loop] <= 0.001 or loop == len(er)-1:
            break;
        loop = loop +1

    plt.xlabel('iterations')
    plt.ylabel('Error')       
    plt.plot(er)
    return t

def hyp(x,t):
    h = 1.0/(1.0 + np.exp(-(np.matrix(t)).T*(np.matrix(x)).T))
    return h

def cost(x,y,theta):
    m = len(x)
    x = np.concatenate((np.ones((m,1)),x,),axis = 1)
    h = np.array(hyp(x,theta)).T
    y = y.reshape((m,1))
    c = 0.0
    c = (-1.0/m)*np.sum(y*np.log(h) + (1-y)*np.log(1-h))
    return np.nan_to_num(c)
    
print "Applying Gradient descent... "
theta_est = GD(train_data01,train_label01,theta)
print "Finding error using cost function..."
error_train = cost(train_data01,train_label01,theta_est)
print "Training Error is = ",error_train

elapsed_time = time.time() - start_time
print "Elapsed simulation time = ",elapsed_time

x = copy.copy(test_data01)
xc = np.concatenate((np.ones((len(x),1)),x,),axis = 1)
hypo = hyp(xc,theta_est)

hypo_01 = copy.copy(hypo)
hypo_01[hypo_01[0,:]>0.5] = 1
hypo_01[hypo_01[0,:]<0.5] = 0
labels = copy.copy(test_label01)
labels = labels.reshape((1,2115))
s = np.sum(labels==hypo_01)
accuracy = np.float(np.float(s)/2115.0)
print "Accuracy on test data = ", accuracy
