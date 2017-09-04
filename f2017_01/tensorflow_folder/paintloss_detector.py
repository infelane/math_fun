"""
Starting from an image, load the network and generate the output
"""
import sys, os

import numpy as np
import tensorflow as tf

folder_loc = '/home/lameeus/Documents/Link to Python/2016_November/PhD/packages'
cmd_subfolder = os.path.realpath(folder_loc)
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
from training_data import Training_data

h0 = 0
h1 = 1945   # 1945
w0 = 0
w1 = 1218   # 1218
images_set = 'beard'

h0 = 0
h1 = 1401   # 1945
w0 = 0
w1 = 2101   # 1218
images_set = 'hand'

ext_out = 0
training_data = Training_data(new=True, amount=2, ext_out=ext_out, images_set=images_set)

foo = training_data.rect_patch(0, 10, 0, 10)

print(np.shape(foo[0].eval()))

def update_progress(workdone):
    print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)

output_map = np.zeros(shape=(h1, w1))

import main2
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    network, network_group = main2.main()
    
    # Replace network!
    network_group

    init = tf.global_variables_initializer()
    sess.run(init)

    # network.clean_init(sess)

    for h_i in range(h0, h1):
        foo = training_data.rect_patch(h_i, h_i+1, w0, w1)
        bar = foo[0].eval()
        bar = np.reshape(bar, newshape=(-1, 3, 3, 7))+0.5

        network.set_dict(bar)

        # predictions = np.asarray(network.predictions(sess))

        feed_dict = {network_group.x: bar}
        predictions = network_group.get_output(feed_dict)

        output_map[h_i, :] = predictions[:, 0, 0, 0]

        # sys.stdout.write("\r%d%%" % int(h_i/h1*100))
        update_progress(h_i/h1)
        sys.stdout.flush()

import matplotlib.pyplot as plt
print('\n')
print(np.min(output_map))
print(np.max(output_map))

from PIL import Image
im = Image.open('/scratch/lameeus/data/altarpiece_close_up/beard_updated/rgb_cleaned.tif')

imarray = np.asarray(im)

input_map = imarray[h0: h1, w0: w1]

import scipy.misc
output_file = 'outfile.tif'
scipy.misc.toimage(output_map, cmin=0.0, cmax=1.0).save('/scratch/lameeus/data/tensorflow/results/' + output_file)

plt.subplot(1, 2, 1)
plt.imshow(input_map, vmin=0, vmax=1)
plt.gray()

plt.subplot(1, 2, 2)
plt.imshow(output_map, vmin=0, vmax=1)
plt.gray()
plt.show()
