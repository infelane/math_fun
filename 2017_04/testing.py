"""Color-Based Segmentation Using the L*a*b* Color Space"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.misc



# hand
input_1 = "/scratch/lameeus/data/altarpiece_close_up/finger/hand_cleaned.tif"
input_2 = "/scratch/lameeus/data/altarpiece_close_up/finger/hand_rgb.tif"
input_3 = "/scratch/lameeus/data/altarpiece_close_up/finger/hand_ir.tif"
output_1 = "/scratch/lameeus/data/altarpiece_close_up/finger/ground_truth.tif"

# # Zachary
# input_1 = "/scratch/lameeus/data/altarpiece_close_up/beard_updated/rgb_cleaned.tif"
# input_2 = "/scratch/lameeus/data/altarpiece_close_up/beard_updated/rgb.tif"
# input_3 = "/scratch/lameeus/data/altarpiece_close_up/beard_updated/ir_non_refl.tif"
# output_1 = "/scratch/lameeus/data/altarpiece_close_up/beard_updated/ground_truth.tif"

def open_image(arr, resize=1.0):
    arr1 = Image.open(arr)
    # resizing
    arr2 = scipy.misc.imresize(arr1, resize, interp='bicubic')
    arr3 = np.asarray(arr2) / 255
    return arr3


im_in_1 = open_image(input_1)

plt.subplot('331')
plt.imshow(im_in_1)


from skimage import io, color
rgb = io.imread(input_1)
lab = color.rgb2lab(rgb)



print(np.shape(lab))

a = np.mean(lab, axis=(0, 1))
b = np.std(lab, axis=(0, 1))

print(a)
print(b)


plt.subplot('332')
plt.imshow(((lab - a)/(10*b) + 0.5), vmin=0.0, vmax=1.0) # [..., 0]


plt.subplot('334')
plt.imshow(lab[..., 0])

plt.subplot('335')
plt.imshow(lab[..., 1])

plt.subplot('336')
plt.imshow(lab[..., 2])


im_in_1_flat = np.reshape(im_in_1, newshape=(-1, 3))

from matplotlib.mlab import PCA

results = PCA(im_in_1_flat)

# c = np.reshape(results.a, newshape=np.shape(im_in_1))

c = results.project(im_in_1_flat)
c = np.reshape(c, newshape=np.shape(im_in_1))
# results.center(im_in_1_flat)

# print(results.a)

plt.subplot('333')
a = np.mean(c, axis=(0, 1))
b = np.std(c, axis=(0, 1))
plt.imshow( ((c - a)/(10*b) + 0.5), vmin=0.0, vmax=1.0)
plt.title('PCA')

plt.subplot('337')
plt.imshow(c[..., 0])

plt.subplot('338')
plt.imshow(-c[..., 1])

plt.subplot('339')
plt.imshow(-c[..., 2])

n_colors = 6

shape_im = np.shape(im_in_1)



import numpy.random as random
random.seed(123)

r = 1


# chromatic components
a = lab[:, :, 1]
b = lab[:, :, 2]
color_markers = np.zeros((n_colors, 2));

for i_color in range(n_colors):
    h_i = int(random.random() * shape_im[0])
    w_i = int(random.random() * shape_im[1])
    color_markers[i_color, :] = np.mean(lab[h_i - r: h_i + r, w_i - r: w_i + r, 1:3], axis = (0, 1))

print(color_markers)

distance = np.zeros(shape = list(shape_im[0 :2]) + [n_colors])
print(np.shape(distance))

for i_color in range(n_colors):
    distance[:, :, i_color] = np.sqrt(np.square(a - color_markers[i_color, 0]) + \
                              np.square(b - color_markers[i_color, 1]))
    
label = np.argmin(distance, axis = 2)



for i_color in range(n_colors):
    result0 = np.copy(im_in_1)
    
    result0[label != i_color, :] = 0.0
    
    plt.subplot('33' + str(i_color+1))
    plt.imshow(result0)

plt.subplot('337')

lab_chroma = np.copy(lab)
# set intensity the same
lab_chroma[..., 0] = np.mean(lab_chroma[..., 0])

lab_chroma[..., 1] = lab_chroma[..., 1]*2
lab_chroma[..., 2] = lab_chroma[..., 2]*2
rgb_chroma = color.lab2rgb(lab_chroma)

plt.imshow(rgb_chroma)
    
# Show results
plt.show()
