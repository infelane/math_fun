# File consisting of multiple handy methods

import numpy as np


# Generate a .txt file with boolean values
def gen_bool_txt(row):

	from numpy import random as random

	my_seed = 3
	random.seed(my_seed)

	save_name = 'gen_input.txt'
	col = 2

	values = np.zeros([row, col])

	for i in range(0, row):
		for j in range(0, col):
			values[i, j] = random.choice([1, 0])

	csv_saver(values, save_name)


def csv_saver(array, name):
	np.savetxt(name, array, delimiter=',')


def xor(input_val):
	a = np.sum(input_val, axis=1)
	output_val = 1*(a == 1)  # The 1* is for converting to integers
	return output_val
