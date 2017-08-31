import numpy as np
import cv2
from skimage.color import colorconv
import matplotlib.pyplot as plt


def n_to_rgb(x, with_col = True, with_sat = False, with_lum = False):
    shape = np.shape(x)
    shape_rgb = [a for a in shape]
    shape_rgb[-1] = 3
    # rgb = np.empty(shape = shape_rgb)
    
    n_col = gen_n_col(shape[-1])
    
    x_arg = np.argmax(x, axis = -1)
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
    
    # print(foo)
    # print(np.shape(foo))
    
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
    if mask is None:
        plt.imshow(rgb)
    else:
        rgb_mask = rgb[...]
        rgb_mask[mask, :] = 0.5
        plt.imshow(rgb_mask)
    
    if title:
        plt.title(title)
