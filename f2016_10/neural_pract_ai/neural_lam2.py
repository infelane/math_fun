# Second program to test neural networks
# Recovering image when passing through opaque glass

import Image
import numpy as np
import matplotlib.pyplot as plt		# Advanced plotter

import lam_neural_general as lam

file_folder = "figs2/"
filename_pattern = ['blur1.jpg', 'blur2.jpg', 'blur3.jpg', 'blur4.jpg', 'blur5.jpg', 'blur6.jpg']
filename_target = 'target.jpg'



size = len(filename_pattern)
print(size)
mat_pattern = []
for i in range(0, size):
	filename = filename_pattern[i]
	image_orig = Image.open(file_folder + filename)
	# lam.showImage(image_orig)
	mat_pattern.append(lam.im2mat(image_orig))
mat_pattern = np.array(mat_pattern)                         # convert to numpy array
image_target = Image.open(file_folder + filename_target)    # Load image of target
mat_target = lam.im2mat(image_target)

fig, (ax1) = plt.subplots(1, 1) # opens a new figure
ax1.set_title('Original')
ax1.imshow(image_target)
fig.show()

# Crop
xBegin = 500
yBegin = 500
xEnd = 1000
yEnd = 1000
mat_pattern =  mat_pattern[:, xBegin:xEnd, yBegin:yEnd]
mat_target =  mat_target[xBegin:xEnd, yBegin:yEnd]
print("Size of input: {}".format(np.shape(mat_pattern)))

mat_gen = lam.learn1(mat_pattern, mat_target)
