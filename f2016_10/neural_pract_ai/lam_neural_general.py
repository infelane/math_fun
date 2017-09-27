import numpy as np
import neurolab as nl  				# Neural Network library
import Image
import matplotlib.pyplot as plt		# Advanced plotter
from numpy import random as random



# prints the image by opening a new tab
def show_image(args):
	args.show()


# Converts image to 2D array
def im2mat(image):
	return np.asarray(image.convert('L'))


def print_matrix(matrix):
	image = Image.fromarray(matrix)
	show_image(image)


# single column
def learn1(pattern, target):
	my_seed = 3
	random.seed(my_seed)

	# Convert matrix into array
	[pics, rows, cols] = np.shape(pattern)
	pixels = rows * cols
	P = np.transpose(((pattern[:, :, :]).reshape(pics, pixels, 1))[:, :, 0])
	T = (target.reshape(pixels, 1))
	print("Size of reshape: {}".format(np.shape(P)))
	print("Size of reshape: {}".format(np.shape(T)))

	idx = np.arange(pixels)
	np.random.shuffle(idx)
	frac_train = 0.01
	lim_train = int(pixels* frac_train);

	PN = (P - np.mean(P, axis= 0))/ np.std(P, axis= 0)  # mean=0 and variance=1

	# TODO 
	P_minmax =  np.transpose([np.min(PN, axis= 0), np.max(PN, axis= 0)])
	P_minmax = [[-0.77, 13]]

	print(PN[0,: ])

	"""
	col1 = PN[:,1]
	#print(col1)
	print(np.mean(col1))
	print(np.std(col1))
	"""

	neurons = [3, 3, 1]             # Amount of nodes in each layer
	transSingle = nl.trans.PureLin()
	trans = [transSingle, transSingle, transSingle]
	#Base for neural network
	net = nl.net.newff(P_minmax*pics, neurons, trans)
	net.trainf = nl.train.train_gd	# Change training func, now gradient descend
	net.errorf = nl.error.MSE()  	# which error function
	net.init()

	epochs = 100
	lr = 0.0001


	P_TRN = PN[idx[:lim_train], :]
	T_TRN = T[idx[:lim_train], :]

	P_val = PN[:, :]
	T_des = T[:, :]

	O_TRN = net.train(P_TRN, T_TRN, epochs=epochs, show=10, goal=0, lr=lr, adapt=True)
	T_gen = net.sim(P_val)  # Generate Target with the learning

	mat_gen = np.reshape(T_gen,[rows, cols])
	mat_des = np.reshape(T_des,[rows, cols])

	imageGen = Image.fromarray(mat_gen)
	imageDes = Image.fromarray(mat_des+0.0001) # TODO CLEAN UP CODE, bug here


	fig, (ax1, ax2) = plt.subplots(1, 2) # opens a new figure
	ax1.set_title('Generated figure')
	ax2.set_title('Desired one')
	ax1.imshow(imageGen)
	ax2.imshow(imageDes)

	plt.show()

	return mat_gen
	# image = Posting.fromarray(matrix1)
	# imgplot = plt.imshow(image)
