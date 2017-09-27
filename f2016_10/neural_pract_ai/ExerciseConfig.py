# -*- coding: utf-8 -*-
""" 
Practical Session: Neural Networks
==================================
Neural network with 1 hidden layer
Set the variable "my_seed" with an interger number.
"""
import neurolab as nl  		# Neural Network library
import numpy as np 
from numpy import random as random

# set up the pseudo-random generation 
my_seed = 3  # set an interger number > 0
random.seed(my_seed)

# Reading and normalizing the data
diabetes = np.genfromtxt('pima-indians-diabetes_data.txt', delimiter=',') # Read the database
n = len(diabetes); n_inputs=8				# Observations and number of inputs
P = diabetes[ :,:-1]						# Patterns (input data)
T = (diabetes[ :,n_inputs]).reshape(n,1)	# Target
# Normalizing the patterns
PN = (P - np.mean(P)) / np.std(P)			# mean=0 and variance=1 

## Split up the input data into Training (TRN) / Validation (VAL)/ Test (TST) sets
frac_train = 0.72;    # Percentage for training set
frac_val = 0.14;      # Percentage for validation set.
idx = np.arange(n)
#np.random.shuffle(idx)
lim_train=int(n*frac_train); lim_val=int(n*(frac_train+frac_val));
P_TRN = PN[idx[:lim_train]]
T_TRN =  T[idx[:lim_train]]
P_VAL = PN[idx[lim_train:lim_val]]
T_VAL =  T[idx[lim_train:lim_val]]
P_TST = PN[idx[lim_val:]] 
T_TST =  T[idx[lim_val:]] 

# Neural network parameters
epochs = 1000;          # Iterations (Number of train epochs)
hidden_nodes = 10;		# Number of nodes 
lr = 0.005;              # Learning rate 
show = 5;				# Print period
mse = nl.error.MSE() 	# Mean Square Error
 
#P_TRN = np.genfromtxt('P_TRN.txt', delimiter=',') 
#P_VAL = np.genfromtxt('P_VAL.txt', delimiter=',')
#P_TST = np.genfromtxt('P_TST.txt', delimiter=',') 
#T_TRN = np.genfromtxt('T_TRN.txt', delimiter=',') 
#T_TRN=T_TRN.reshape(len(T_TRN),1)
#T_VAL = np.genfromtxt('T_VAL.txt', delimiter=',')
#T_VAL=T_VAL.reshape(len(T_VAL),1);  
#T_TST = np.genfromtxt('T_TST.txt', delimiter=',') 
#T_TST=T_TST.reshape(len(T_TST),1); 

P_minmax=[[np.min(P_TRN), np.max(P_TRN)]] # min and max values
# Create a feedforward network with 2 layers and 'hidden_nodes' hidden nodes
net = nl.net.newff(P_minmax*n_inputs,[hidden_nodes, 1],[nl.trans.LogSig(),nl.trans.PureLin()])
net.trainf = nl.train.train_gd	# Traning function
net.errorf = nl.error.MSE()  	# Perform function
net.init()

# Train the network 
perf=np.zeros((epochs, 3))  
for i in np.arange(epochs):
	O_TRN = net.train(P_TRN, T_TRN, epochs=0, show=2, goal=0, lr=lr, adapt=True)
	O_VAL = net.sim(P_VAL)
	O_TST = net.sim(P_TST)
	perf[i,0] = O_TRN[-1]
	perf[i,1] = mse(T_VAL, O_VAL)
	perf[i,2] = mse(T_TST, O_TST)
	if (i+1)%show==0:
		print("epoch {} TRN:{} VAL:{} TST:{}".format(i+1,perf[i,0], perf[i,1], perf[i,2]))
mse_val = perf[:,1]
title="Best validation performance is {} at epoch {}\n".format( np.min(mse_val),np.argmin(mse_val) )
# Perform plot
import pylab as pl
pl.plot(perf[:,0],'-' ,perf[:,1], '--', perf[:,2],':')
pl.axvline(np.argmin(mse_val),color = 'y',ls='dashed')
pl.xlabel('{} Epochs'.format(epochs))
pl.ylabel('Mean Squared Error (mse)')
pl.title(title)
pl.legend(['train', 'validation','test'])
pl.show()
