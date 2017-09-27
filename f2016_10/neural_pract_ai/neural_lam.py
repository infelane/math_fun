# -*- coding: utf-8 -*-
""" 
By Laurens Meeus
Practical Session: Neural Networks
==================================
Neural network with 2 hidden layers
Set the variable "my_seed" with an interger number.
Hint: You can try with a higher number of neurons. 
"""
import neurolab as nl  				# Neural Network library
import numpy as np 					
from numpy import random as random
import matplotlib.pyplot as plt		# Advanced plotter

#from config import *
from config2 import *



# set up the pseudo-random generation 
my_seed = 4  # set an interger number > 0
random.seed(my_seed) 

# Reading and normalizing the data
diabetes = np.genfromtxt('pima-indians-diabetes_data.txt', delimiter=',') # Read the database

[n, n_inputs] = np.shape(diabetes)  # Observations and number of inputs
n_inputs = n_inputs-1               # Don't include the target
#n = len(diabetes); n_inputs=8
P = diabetes[ :,:-1]				      # Patterns (input data), first 8 colums
T = (diabetes[ :,n_inputs]).reshape(n,1)	# Target
# Standardizing the patterns
PN = (P - np.mean(P)) / np.std(P)			# mean=0 and variance=1 
## Split up the input matrix into Training (TRN)/Validation (VAL) / Test sets.
frac_train = 0.6; 	# Percentage for training set.
frac_val = 0.2;      	# Percentage for validation set.
idx = np.arange(n)
np.random.shuffle(idx)
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

# plot settings for the correlation
# DONE
def settings1(ax):
	labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
	range_label = np.arange(0.5,9.0,1)
	
	ax.set_xticklabels(labels)
	ax.set_yticklabels(labels)

	plt.sca(ax)
	plt.xticks(range_label)
	plt.yticks(range_label)	
	
# Improvised breakpoint
def wait():
	wait = input("PRESS ENTER TO CONTINUE.")

# Actually correlation matrix, visualization of relationship between multiple inputs
# DONE
def showCovariance(data):
	size = data.shape;
	print(size)
	corr = np.corrcoef(data,rowvar=False) #correlation of the different columns, the linear relationship
	print(corr)

	cm = 'PiYG'
			
	fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True) # opens a new figure
	ax1.set_title('Correlation matrix')
	ax2.set_title('Without diagonal')

	im1 = ax1.pcolor(corr, cmap = cm, vmin=-1,vmax = 1) 	# The colormap plot
	plt.colorbar(im1, ax = ax1)
	settings1(ax1)
	
	# removes the diagonal and checks the max absolute corr.
	for i in range(0,size[1]):
		corr[i,i] = 0;
	max_corr = np.max([-np.min(corr),np.max(corr)])
	
	im2 = ax2.pcolor(corr, cmap = cm, vmin=-max_corr,vmax = max_corr)	
	plt.colorbar(im2, ax = ax2)
	settings1(ax2)
	
	plt.show(block = False) #Shows the figure, but keeps program running
	
		
showCovariance(diabetes)

# TODO asdfasdf

mse = nl.error.MSE() 	# Mean Square Error

# Training loop
results = np.zeros((m-1, 2)) 
for j in range(2,m+1,1):
	print ('neurons: {}'.format(j)) 
	results[(j-2),1]=j
	# Create a feedforward network with 2 hidden layers and 'j' hidden nodes
	# (minmax for each column, amount of neurons at each layer, List of activation function for each layer


	net = nl.net.newff(P_minmax*n_inputs,[j, j, 1],[nl.trans.TanSig(),nl.trans.TanSig(),nl.trans.TanSig()])
	net.trainf = nl.train.train_gd	# Change training func, now gradient descend
	net.errorf = nl.error.MSE()  	# which error function
	# Choosing the best performance between n simulations
	for i in range(0,n,1): 
		
		net.init() # resets begin values
		perf=np.zeros((epochs, 3)) 
		for ii in np.arange(epochs):
			#net.train(P_TRN, T_TRN, epochs=epochs, show=show, goal=0, lr=lr, adapt=True)
			O_TRN = net.train(P_TRN, T_TRN, epochs=1, show=0, goal=0, lr=lr, adapt=True)
			O_VAL = net.sim(P_VAL) # Generate Target with the learning
			O_TST = net.sim(P_TST)
			
		
			
			perf[ii,0] = O_TRN[-1]
			perf[ii,1] = mse(T_VAL, O_VAL)	#Compare how target fits generated target.
			perf[ii,2] = mse(T_TST, O_TST)
			if (ii+1)%show==0:
				print("epoch {} TRN:{} VAL:{} TST:{}".format(ii+1,perf[ii,0], perf[ii,1], perf[ii,2]))

		# plot of the learning
		mse_val = perf[:,1] 	# MSE of the validated training
		title="Best validation performance is {} at epoch {}\n".format( np.min(mse_val),np.argmin(mse_val) )

		plt.figure()
		plt.plot(perf[:,0],'-' ,perf[:,1], '--', perf[:,2],':')
		plt.axvline(np.argmin(mse_val),color = 'y',ls='dashed')
		plt.xlabel('{} Epochs'.format(epochs))
		plt.ylabel('Mean Squared Error (mse)')
		plt.title(title)
		plt.legend(['train', 'validation','test'])

		
		#net.init()
		#error = net.train(P_TRN, T_TRN, epochs=epochs, show=show, goal=0, lr=lr, adapt=True)
		
		error = perf[:,0]
		
		#error = perf[:,0]
		
		#plot validation
		
		
		
		if i == 1:
			results[j-2,0]=error[-1]
		elif error[-1] < results[j-2,0]:
			results[j-2,0]=error[-1]
	# This is the best
	print("The best performance ({} neurons) is : {}".format(j, results[j-2,0]))

# Performance plot
plt.figure()
plt.plot(results[:,1], results[:,0])
plt.xlabel('Number of neurons')
plt.ylabel('Mean Squared Error on training (mse)')
 
plt.show(block = True) # as long as some plots are showed


