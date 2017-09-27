# Easy test for neural networks (general)

import numpy as np
import useful1
import lam_neural_general3 as lam

rows = 200
useful1.gen_bool_txt(rows)


nameFun = 'XOR'

input_name = 'gen_input.txt'
input_val = np.genfromtxt(input_name, delimiter=',')  # Read the database

noise = np.random.normal(0, 0.3, rows*3)   # gaussian noise centered around 0, with width 0.1
noise = noise.reshape([rows, 3])
input_val_noise = input_val + noise[:, 0:-1]

print(input_val_noise)

output_val = useful1.xor(input_val)
useful1.csv_saver(output_val, 'gen_output.txt')

output_val = output_val + noise[:, -1]

lam.lam_net(input_val_noise, output_val)
