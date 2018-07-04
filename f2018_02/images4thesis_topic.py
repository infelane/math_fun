import matplotlib.pyplot as plt
import numpy as np

folder = 'C:/Users/Laurens-W10/Downloads/'
rgb = plt.imread(folder + 'rgb.bmp')
ir = plt.imread(folder + 'ir.bmp')
irr = plt.imread(folder + 'irr.bmp')
xr = plt.imread(folder + 'xr.bmp')


def cropper(a):
    h0 = 450
    h1 = 900
    w0 = 500
    w1 = 900

    return a[h0:h1, w0:w1, :]

rgb = cropper(rgb)
irr = cropper(irr)
xr = cropper(xr)

plt.imsave(folder + 'rgb_crop.png', rgb)

rgb_single = np.mean(rgb, axis=-1)/255.
irr_single = np.mean(irr, axis=-1) /255.
xr_single = np.mean(xr, axis=-1) /255.

def normalizer(a, width = 2, off = 0.):
    """

    :param a:
    :param width: width times the std
    :return:
    """
    b = np.mean(rgb_single)
    c = np.std(rgb_single)
    print(b)
    print(c)

    d = (a - b)/(c*2 * width) + 0.5 + off # normalization

    d[d < 0.] = 0.
    d[d > 1.] = 1.

    return d


rgb_single = normalizer(rgb_single, 1, off = -0.2)
irr_single = normalizer(irr_single, 1.5, off = -0.2)
xr_single = normalizer(xr_single, 0.5, off = -0.2)

new = np.stack([rgb_single, xr_single, irr_single], axis=-1) # best

plt.subplot(2, 2, 1)
plt.imshow(rgb_single)
plt.subplot(2, 2, 2)
if 0:
    plt.imshow(ir)
    plt.subplot(2, 2, 3)
plt.imshow(irr_single)
plt.subplot(2, 2, 3)
plt.imshow(xr_single)
plt.subplot(2, 2, 4)
plt.imshow(new)


def save_settings(path, a, cmap = 'gray' ):
    plt.imsave(path, a, vmin=0., vmax=1., cmap=cmap)

save_settings(folder + 'rgb_single.png', rgb_single)
save_settings(folder + 'irr_single.png', irr_single)
save_settings(folder + 'xr_single.png', xr_single)

plt.imsave(folder + 'all.png', new)

plt.show()
