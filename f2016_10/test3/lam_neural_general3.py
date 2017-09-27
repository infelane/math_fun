import numpy as np
import neurolab as nl  # Neural Network library
import Image
import matplotlib.pyplot as plt  # Advanced plotter
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
	lim_train = int(pixels * frac_train);

	PN = preproc(P)

	# TODO 
	P_minmax = np.transpose([np.min(PN, axis=0), np.max(PN, axis=0)])
	P_minmax = [[-0.77, 13]]

	print(PN[0, :])

	"""
	col1 = PN[:,1]
	#print(col1)
	print(np.mean(col1))
	print(np.std(col1))
	"""

	neurons = [3, 3, 1]  # Amount of nodes in each layer
	transSingle = nl.trans.PureLin()
	trans = [transSingle, transSingle, transSingle]
	# Base for neural network
	net = nl.net.newff(P_minmax * pics, neurons, trans)
	net.trainf = nl.train.train_gd  # Change training func, now gradient descend
	net.errorf = nl.error.MSE()  # which error function
	net.init()

	epochs = 100
	lr = 0.0001

	P_TRN = PN[idx[:lim_train], :]
	T_TRN = T[idx[:lim_train], :]

	P_val = PN[:, :]
	T_des = T[:, :]

	O_TRN = net.train(P_TRN, T_TRN, epochs=epochs, show=10, goal=0, lr=lr, adapt=True)
	T_gen = net.sim(P_val)  # Generate Target with the learning

	mat_gen = np.reshape(T_gen, [rows, cols])
	mat_des = np.reshape(T_des, [rows, cols])

	imageGen = Image.fromarray(mat_gen)
	imageDes = Image.fromarray(mat_des + 0.0001)  # TODO CLEAN UP CODE, bug here

	fig, (ax1, ax2) = plt.subplots(1, 2)  # opens a new figure
	ax1.set_title('Generated figure')
	ax2.set_title('Desired one')
	ax1.imshow(imageGen)
	ax2.imshow(imageDes)

	plt.show()

	return mat_gen


# image = Posting.fromarray(matrix1)
# imgplot = plt.imshow(image)


def lam_net(input_val,output_val):

	#making sure shape is good
	[rows, cols] =  np.shape(input_val)
	input_val = input_val.reshape(rows,cols)
	output_val = output_val.reshape(rows,1)













	lam_net_class = Lam_net_class()
	net = lam_net_class.get_net()

	p_n = preproc(input_val)


	#net = lam_net_class.get_net()

	print(np.shape(output_val))
	print(np.shape(p_n))



	# Create network with 2 layers and random initialized
	net = nl.net.newff([[-1, 1], [-1, 1]], [3, 3, 1])
	net.trainf = nl.train.train_gd  # Change training func, now gradient descend, which means you can give the option 'lr'

	# Train network
	error = net.train(p_n, output_val, epochs=500, show=50, goal=0, lr = 0.01)

	a = net.sim(p_n)



	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.pyplot as plt

	def randrange(n, vmin, vmax):
		return (vmax - vmin) * np.random.rand(n) + vmin

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')


	ax.scatter(p_n[:,0],p_n[:,1],a,c='r', marker='*')

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	plt.show()



	# plt.Axes3D.scatter(xs, ys, zs=0, zdir='z', s=20, c=None, depthshade=True, *args, **kwargs)
	plt.Axes3D.scatter(p_n[:,0], p_n[:,1], a, zdir='z', s=20, c=None, depthshade=True, *args, **kwargs)

	print(error)

	plt.plot(error, '*')
	plt.show()
	plt.legend(['error of network'])



	net = nl.net.newff([[-7, 7]], [5, 1])

	net.train(p_n, output_val, epochs=100, show=10, goal=0, lr=0.0001, adapt=True)

	print(net)

	#P_TRN = PN[idx[:lim_train], :]
	#T_TRN = T[idx[:lim_train], :]

	#P_val = PN[:, :]
	#T_des = T[:, :]

	P_TRN = input_val
	T_TRN = output_val

	#lam_net_class.train(input_val,output_val)
	#lam_net_class.train(P_TRN, T_TRN)




# For faster learning it is proven to be advantageous to make the mean zero and normalize the variance
def preproc(pattern):
	p_n = (pattern - np.mean(pattern, axis=0)) / np.std(pattern, axis=0)  # mean=0 and variance=1
	return p_n


# Will be initialized with
class Lam_net_class():

	neurons = []    # Amount of nodes in each layer
	epochs = []     # amount of improvement steps
	lr = []         # learning rate
	net = []       # base for neural network

	neurons = [2, 1]
	transSingle = nl.trans.PureLin()
	trans = [transSingle, transSingle]
	# Base for neural network

	P_min_max = [[-1, 1]] # TODO understand what this means

	def __init__(self):

		self.epochs = 100
		self.lr = 0.0001

		net_here = nl.net.newff(self.P_min_max, self.neurons, self.trans)
		net_here.trainf = nl.train.train_gd  # Change training func, now gradient descend
		net_here.errorf = nl.error.MSE()  # which error function
		net_here.init()

		self.net = net_here


	def get_net (self):
		return self.net

	def train (self, pattern, target):
		self.net.train(pattern, target, epochs=self.epochs, show=10, goal=0, lr=self.lr, adapt=True)
