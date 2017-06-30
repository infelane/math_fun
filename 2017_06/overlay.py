"""
Generates overlay between two predictions
"""
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np


folder = '/ipi/private/lameeus/data/lamb/output/'
net_a = Image.open(folder + 'zach_gen.png')
folder = '/scratch/lameeus/data/altarpiece_close_up/beard_updated/'
ref_a = Image.open(folder + 'ground_truth.tif')
clean = Image.open(folder + 'rgb_cleaned.tif')

net_a = np.asarray(net_a)
ref_a = np.asarray(ref_a)
clean = np.asarray(clean)

# Make background image more dull
clean =  ((clean - 127.5)*0.5+ 127.5)
clean[clean<0]=0
clean[clean>255]=255
clean = clean.astype(np.uint8)

# overlay = np.zeros(shape)
overlay = np.copy(clean)

blue = np.asarray([0, 0, 255])
yellow = np.asarray([255, 255, 0])
green = np.asarray([0, 255, 0])
red = np.asarray([255, 0, 0])

ref_loss = ref_a[:, :, 0] == 255
net_loss = net_a[:, :] == 255

overlay[~net_loss & ref_loss] = green   # missed regions
overlay[net_loss & ~ref_loss] = red     # over classification
overlay[ref_loss & net_loss] = blue     # correct

imgplot = plt.imshow(net_a)
plt.figure()
imgplot = plt.imshow(overlay)
plt.show()

img = Image.fromarray(overlay)
img.save("overlay.tif")
