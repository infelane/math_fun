import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import timeit


def tophat_tf(im, ext = None, kernel = None):
    if kernel is None:
        width_ext = 1 + 2 * ext
        kernel = np.ones(shape=(width_ext, width_ext))

    im_tophat = im - conv(im, kernel)
    
    return im_tophat


def bothat_tf(im, ext = None, kernel = None):
    if kernel is None:
        width_ext = 1 + 2 * ext
        kernel = np.ones(shape=(width_ext, width_ext))
    
    im_tophat = conv(im, kernel) - im
    
    return im_tophat

def step(x, thresh):
    return 1 * (x > thresh)

def extend(array, ext):
    width_ext = 1 + 2 * ext
    shape = np.shape(array)
    height, width = shape
    array_ext = np.zeros(shape=(height + width_ext, width + width_ext))
    array_ext[ext:height + ext, ext:width + ext] = array
    return array_ext

def relu(x):
    return np.maximum(x, 0, x)

def conv(array, kernel):
    # Normalization of the kernel
    kernel = kernel/np.sum(kernel)
    
    width_ext = np.shape(kernel)[0]
    ext = int((width_ext -1)/2)
    shape = np.shape(array)
    height, width = shape
    
    array_ext = extend(array, ext)
    
    next_im = np.zeros(shape=shape)
    
    for h_i in range(width_ext):
        for w_i in range(width_ext):
            next_im[...] += kernel[h_i, w_i]*array_ext[h_i : height+h_i, w_i: width + w_i]
    
    return next_im

def plotter(im, ax):
    return ax.imshow(im, vmin=-1, vmax=1., cmap = 'seismic')
    
    
def gaussian_kernel(size, size_y=None):
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    x, y = np.mgrid[-size:size+1, -size_y:size_y+1]
    g = np.exp(-(x**2/float(size)+y**2/float(size_y)))
    return g / g.sum()

    
def main():
    t_start = timeit.timeit()
    
    im_index = [2, 3, 7]

    f, axarr = plt.subplots(3, 3)

    import matplotlib

    def move_figure(f, x, y):
        """Move figure's upper left corner to pixel (x, y)"""
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
        f.set_size_inches(12.0, 20.0, forward=True)


    move_figure(f, 1950, 0)
    # f.set_size_inches(50, 10.5)

    # f.canvas.manager.window.move(0, 0)
    # mngr = plt.get_current_fig_manager()
    # mngr.window.setGeometry(50, 100, 640, 545)
    
    for index in range(3):
    
        folder = '/home/lameeus/data/xray/cracks/images/'
        name = folder + '{}.png'.format(im_index[index])
        # Opening image
        im_inp_float = np.asarray(Image.open(name)) / 255
    
        # grey = np.mean(im_inp_float, axis = 2)
        im = im_inp_float[... ,0]
        shape = np.shape(im)
        height, width = shape

        a = axarr[index, 0]
        
        """ Normalization """
        grey_avg = np.mean(im)
        grey_std = np.std(im)
        
        print(grey_std)

        # Local averaging
        im = tophat_tf(im, ext=20)*2
        # # General averaging
        # im = (im - grey_avg)*2
        
        plot = plotter(im, a)
        
        """ Remove light regions """

        im = -relu(-im)

        a = axarr[index, 1]
        plotter(im, a)
        
        """ Edge detection """

        im = tophat_tf(im, ext = 3)
        
        a = axarr[index, 2]
        plotter(im, a)

        # TODO here
        
        # print(grey_avg)
        #
        # im = relu(-(im - grey_avg))/grey_std
        
       

      
        
        # background substraction
        # # grey = tophat_tf(grey, ext=10)
        # kernel = gaussian_kernel(10)
        # orig_norm = tophat_tf(im, kernel=kernel)

        # a = axarr[index, 1]
        # plotter(orig_norm, a)
        # plt.show()
        


       
        # plotter(im, a)
        
        # Only dark values
        # orig_norm_relu = -relu(-orig_norm)

        # a = axarr[index, 2]
        # plotter(orig_norm_relu, a)
        
    f.colorbar(plot)
    plt.show()
    
    
    



    foo_array = np.zeros((height, width, 0))
    for color_i in range(3):
        stepsize = 1
        ext = color_i * stepsize + 1
    
        width_ext = 1 + 2 * ext
    
        kernel = np.ones(shape=(width_ext, width_ext))
    
        next_im = conv(grey, kernel).reshape((height, width, 1))
    
        foo_array = np.append(foo_array, next_im, axis=2)

    t_end = timeit.timeit()

    print("elapsed time: {}".format(t_end - t_start))

    # foo_rgb = np.concatenate(foo_array[0], foo_array[0], foo_array[0], axis=2)
    #
    # plt.subplot('311')
    # plt.imshow(grey, vmin= 0.0, vmax=1.0, cmap = 'Greys')
    # plt.colorbar()
    # plt.subplot('312')
    # plt.imshow(foo_array, vmin= 0.0, vmax=1.0)
    # plt.colorbar()
    # plt.subplot('111')

    tophat = foo_array - grey.reshape(height, width, 1)

    # avg =

    tophat_step = step(tophat, thresh=0.05)

    # plt.subplot('311')
    # plotter(tophat_step, a)

    grey_tophat = np.mean(tophat_step, axis=2)

    width_ext = 3
    kernel = np.ones(shape=(width_ext, width_ext))

    foo = conv(grey_tophat, kernel)

    # plt.subplot('312')
    # plotter(foo, a)

    # plt.subplot('311')
    # plt.imshow(next_im, vmin= 0.0, vmax=1.0, cmap = 'Greys')
    # plt.colorbar()
    # plt.show()

    b = step(foo, thresh=0.3)
    
    # plt.subplot('313')
    # plotter(b, a)
    
    # plt.show()


if __name__ == '__main__':
    main()
