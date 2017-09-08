import numpy as np
import cv2
from skimage.color import colorconv
import matplotlib.pyplot as plt


def n_to_rgb(x, with_col = True, with_sat = False, with_lum = False, anno_col = False):
    shape = np.shape(x)

    # rgb = np.empty(shape = shape_rgb)

    x_arg = np.argmax(x, axis=-1)
    if anno_col:
        shape_rgb = [a for a in shape]
        shape_rgb[-1] = 3
        
        red = [1, 0, 0]
        green = [0, 1, 0]
        blue = [0, 0, 1]
        cyan = [0, 1, 1]
        yellow = [1, 1, 0]
        magenta = [1, 0, 1]
        white = [1, 1, 1]
        black = [0, 0, 0]

        rgb = np.zeros(shape_rgb, dtype=float)

        rgb[x_arg == 0, :] = red
        rgb[x_arg == 1, :] = green
        rgb[x_arg == 2, :] = blue
        rgb[x_arg == 3, :] = cyan
        rgb[x_arg == 4, :] = yellow
        rgb[x_arg == 5, :] = magenta
        rgb[x_arg == 6, :] = white
        rgb[x_arg == 7, :] = black
        
        return rgb
        
    else:
    
        n_col = gen_n_col(shape[-1])
        
        x_max = np.max(x, axis = -1)#[x_arg]
        # x_max = x[x_arg]
        # print(np.shape(x_arg))
        # print(np.shape(x_max))
        # for i in range(sha)
        
        # def func(a):
        #     return n_col[a]
        
        if with_col:
            foo = n_col[x_arg]
        else:
            zero = np.zeros(np.shape(x_arg), dtype=int)
            foo = n_col[zero]
        
        if with_sat:
            foo[..., 1] = x_max   # adjust S value to x_p_max
            
        if with_lum:
            foo[..., 2] = 0.5 + 0.5*x_max   # adjust S value to x_p_max
           
        black = np.sum(x, axis = -1) == 0
        foo[black, :] = 0
        
        # return colorsys.hls_to_rgb(foo)
        return colorconv.hsv2rgb(foo)
    

def gen_n_col(n = 1):
    # HLS_tuples = np.array([(x*1.0/n, 0.5, 1.) for x in range(n)])
    HSV_tuples = np.array([(x * 1.0 / n, 1., 1.) for x in range(n)])

    return HSV_tuples


def imshow(rgb, n = 1, mask = None, title = None):
    """
    :param rgb:
    :param n:
    :param mask:
    :return:
    """
    # TODO n, multiple img's
    
    plt.figure()
    
    def plotter(a, b):
        if mask is None:
            plt.imshow(a)
        else:
            rgb_mask = np.zeros(shape = np.shape(a))
            rgb_mask[...] = a
            rgb_mask[mask == 0, :] = 0.5
            plt.imshow(rgb_mask)
    
        if b:
            plt.title(b)
    
    
    if n == 1:
        plotter(rgb, title)
        
    else:
        for i in range(n):
            plt.subplot(1, n, i+1)
            plotter(rgb[i], title[i])


def show_histo(array, show: bool = True):
    a_min = np.min(array)
    a_max = np.max(array)
    
    intens, bins = np.histogram(np.reshape(array, (-1)), bins=50, range=[a_min, a_max])

    bins_center = (bins[0:-1] + bins[1:]) / 2.0

    plt.figure()
    plt.plot(bins_center, intens)
    if show:
        plt.show()
