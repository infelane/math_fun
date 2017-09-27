# -*- coding: utf-8 -*-
""" 
Practical Session: Neural Networks
==================================
Neural network with 2 hidden layers
Set the variable "my_seed" with an interger number.
Hint: You can try with a higher number of neurons. 
"""
import neurolab as nl  				# Neural Network library
import numpy as np 					
from numpy import random as random

# set up the pseudo-random generation 
my_seed = 3  # set an interger number > 0
random.seed(my_seed) 

# Reading and normalizing the data
diabetes = np.genfromtxt('pima-indians-diabetes_data.txt', delimiter=',') # Read the database
n = len(diabetes); n_inputs=8			# Observations and number of inputs
P = diabetes[ :,:-1]				      # Patterns (input data)
T = (diabetes[ :,n_inputs]).reshape(n,1)	# Target
# Standardizing the patterns
PN = (P - np.mean(P)) / np.std(P)			# mean=0 and variance=1 
## Split up the input matrix into Training (TRN)/Validation (VAL) / Test sets.
frac_train = 0.6; 	# Percentage for training set.
frac_val = 0.2;      	# Percentage for validation set.
idx = np.arange(n)
np.random.shuffle(idx)	# Shuffle the indices of all the candidates
lim_train=int(n*frac_train); 
lim_val=int(n*(frac_train+frac_val));
P_TRN = PN[idx[:lim_train]]
T_TRN =  T[idx[:lim_train]]
P_VAL = PN[idx[lim_train:lim_val]]
T_VAL =  T[idx[lim_train:lim_val]]
P_TST = PN[idx[lim_val:]] 
T_TST =  T[idx[lim_val:]] 
# min and max input
P_minmax=[[np.min(P_TRN), np.max(P_TRN)]]
# set up
lr=0.05		# Learning rate
epochs=1000	# Iterations (Number of train epochs)
show= 10	# Print period
m = 10;		# max number of neurons  
# Training loop
results = np.zeros((m-2, 2))     



for j in range(2,m,1):
	print ('neurons: {}'.format(j)) 
	results[(j-2),1]=j
	# Create a feedforward network with 2 hidden layers and 'j' hidden nodes
	net = nl.net.newff(P_minmax*n_inputs,[j, j, 1],[nl.trans.TanSig(),nl.trans.TanSig(),nl.trans.TanSig()])
	net.trainf = nl.train.train_gd	# Change traning func 
	net.errorf = nl.error.MSE()  
	# Choosing the best performance
	for i in range(1,6,1): #redo 6 times
		net.init()
		error = net.train(P_TRN, T_TRN, epochs=epochs, show=show, goal=0, lr=lr, adapt=True)
		
		#plot validation
		
		
		if i == 1:
			results[j-2,0]=error[-1]
		elif error[-1] < results[j-2,0]:
			results[j-2,0]=error[-1]
	print("The best performance ({} neurons) is : {}".format(j, results[j-2,0]))

# Performance plot
import matplotlib.pyplot as plt
plt.plot(results[:8,1], results[:8,0])
plt.xlabel('Number of neurons')
plt.ylabel('Mean Squared Error (mse)')
plt.show()

