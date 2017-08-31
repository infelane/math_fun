"""
Generates overlay between two predictions
"""

import matplotlib.pyplot as plt
# from PIL import Image
import numpy as np

from link_to_soliton.paint_tools import image_tools

# folder = '/ipi/private/lameeus/data/ghent_altar/output/'
# net_a = image_tools.path2im(folder + 'zach_gen.png')
# net_a = image_tools.path2im(folder + 'gen_zach.png')
folder = '/scratch/lameeus/data/tensorflow/results/'
net_a = image_tools.path2im(folder + 'outfile_beard_OLD.tif')
folder = '/scratch/lameeus/data/ghent_altar/altarpiece_close_up/beard_updated/'
ref_a = image_tools.path2im(folder + 'ground_truth.tif')
clean = image_tools.path2im(folder + 'rgb_cleaned.tif')
# folder = '/home/lameeus/data/ghent_altar/output/'
# net_a = image_tools.path2im(folder + 'hand_7in_vhand_vno_clean.tif')
# folder = '/home/lameeus/data/ghent_altar/input/'
# ref_a = image_tools.path2im(folder + '19_annot.tif')
# clean = image_tools.path2im(folder + '19_clean_crop_scale.tif')

# Make background image more dull
clean =  ((clean - 0.5)*0.5+ 0.5)
clean[clean<0]=0
clean[clean>1]=1

# overlay = np.zeros(shape)
overlay = np.copy(clean)

blue = np.asarray([0, 0, 1])
yellow = np.asarray([1, 1, 0])
green = np.asarray([0, 1, 0])
red = np.asarray([1, 0, 0])

if 1:
    net_a = 1-net_a

ref_loss = ref_a[:, :, 0] == 1
net_loss = net_a[:, :] >= 0.5 # 0.05

overlay[~net_loss & ref_loss] = green   # missed regions
overlay[net_loss & ~ref_loss] = red     # over classification
overlay[ref_loss & net_loss] = blue     # correct

plt.figure()
imgplot = plt.imshow(ref_loss, vmin = 0, vmax = 1)
plt.figure()
imgplot = plt.imshow(net_a, vmin = 0, vmax = 1)
plt.figure()
imgplot = plt.imshow(overlay)
plt.show()

folder = '/home/lameeus/data/ghent_altar/overlay/'

image_tools.save_im(overlay, folder + "overlay.tif")
