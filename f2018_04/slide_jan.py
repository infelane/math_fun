import matplotlib.pyplot as plt
import numpy as np

folder = '/scratch/Downloads_local/'
bic = plt.imread(folder + 'bicubic.png')
shear = plt.imread(folder + 'shearlet.png')[..., 0:3]
rcnn = plt.imread(folder + 'rcnn.png')

diff = 0.5 + np.mean(rcnn - shear, axis=2)/255.
diff2 = 0.5 + np.mean(rcnn - bic, axis=2)/255.

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(shear)
plt.subplot(2, 2, 2)
plt.imshow(rcnn)
plt.subplot(2, 2, 3)
plt.imshow(diff2, cmap='seismic')
plt.subplot(2, 2, 4)
plt.imshow(diff, cmap='seismic')

# w, h
crop1_topleft = [450, 100]
crop1_botright = [550, 325]

# good
crop1_topleft = [450, 200]
crop1_botright = [575, 325]

# closer
# crop1_topleft = [450, 250]
# crop1_botright = [525, 325]

def plot4(im1, im2, im3, im4):
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(im1)
    plt.subplot(2, 2, 2)
    plt.imshow(im2)
    plt.subplot(2, 2, 3)
    plt.imshow(im3, cmap='seismic')
    plt.subplot(2, 2, 4)
    plt.imshow(im4, cmap='seismic')

def cropper(im, crop1_topleft, crop1_botright):
    w0, h0 = crop1_topleft
    w1, h1 = crop1_botright
    return im[h0:h1, w0:w1]
    
crop_bic = cropper(bic, crop1_topleft, crop1_botright)
crop_shear = cropper(shear, crop1_topleft, crop1_botright)
crop_rcnn = cropper(rcnn, crop1_topleft, crop1_botright)

plot4(crop_bic, crop_shear, crop_rcnn, diff)

if 1:
    plt.imsave(folder + 'diff.png', diff, cmap='seismic')

    plt.imsave(folder + 'crop_bic.png', crop_bic)
    plt.imsave(folder + 'crop_shear.png', crop_shear)
    plt.imsave(folder + 'crop_rcnn.png', crop_rcnn)

plt.show()