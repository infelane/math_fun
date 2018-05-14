""" for Roman """

import matplotlib.pyplot as plt
import numpy as np

path_im = 'im_in.tif'   # 'im_in.png'
path_out1 = 'im_out1.png'
path_out2 = 'im_out2.png'

anot_color = 'cyan' # 'cyan'

im = plt.imread(path_im)

val_max = np.max(im)
if val_max <= 1.0:
    # float values between 0 and 1
    im_new = im

elif val_max <= 255:
    # integer values between 0 and 255
    im_new = np.divide(im, 255.)

else:
    raise ValueError('TODO: Implement for higher range!')

r0 = np.equal(im_new[..., 0], 0.)
r1 = np.equal(im_new[..., 0], 1.)
g0 = np.equal(im_new[..., 1], 0.)
g1 = np.equal(im_new[..., 1], 1.)
b0 = np.equal(im_new[..., 2], 0.)
b1 = np.equal(im_new[..., 2], 1.)

cyan = [0, 1, 1]
red = [1, 0, 0]

# cyan_map = np.logical_and(np.logical_and(r0, g1), b1)     # did not work (cyan was [0.337, 1, 1]
if anot_color == 'cyan':   # cyan_map
    cyan_map = np.logical_and(b1, g1)
elif anot_color == 'red':
    red_map =  np.logical_and(b0, np.logical_and(r1, g0))

im_new2 = im_new*0.5 + 0.5  # very bright, use 'dark' colors like red, blue, green
im_new_red = np.copy(im_new2)
im_new3 = im_new*0.5    # very dark, use 'light' colors like cyan, magenta and yellow
im_new4 = np.copy(im_new3)

if anot_color == 'cyan':
    im_new2[cyan_map, :] = cyan
    im_new_red[cyan_map, :] = red
    im_new3[cyan_map, :] = cyan
    im_new4[cyan_map, :] = red
elif anot_color == 'red':
    im_new2[red_map, :] = cyan
    im_new_red[red_map, :] = red
    im_new3[red_map, :] = cyan
    im_new4[red_map, :] = red

n_w = 3

plt.subplot(n_w, 2, 1)
plt.imshow(im_new)

plt.subplot(n_w, 2, 2)
if anot_color == 'cyan':
    plt.imshow(cyan_map)
elif anot_color == 'red':
    plt.imshow(red_map)

plt.subplot(n_w, 2, 3)
plt.imshow(im_new2)

plt.subplot(n_w, 2, 4)
plt.imshow(im_new_red)

plt.subplot(n_w, 2, 5)
plt.imshow(im_new3)

plt.subplot(n_w, 2, 6)
plt.imshow(im_new4)


plt.imsave(path_out1 , im_new_red)
plt.imsave(path_out2 , im_new3)
