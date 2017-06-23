import matplotlib.pyplot as plt
# from keras.preprocessing import image
from PIL import Image
import time
import numpy as np
# import skimage
import scipy.ndimage
import tifffile as tiff
import cv2
import pylab


def time_func(func):
    start = time.time()
    a = func()
    end = time.time()
    print('elapsed time: {} s'.format(end - start))
    return a


def path2im(path):
    # scipy.ndimage.imread(path) # DOES not work for 16 bit tif
    # Image.open(path_i) # does not open with 16 bit tif
    im = tiff.imread(path)  # colors stay correct
    # im = cv2.imread(path, -1)  # wrong color conversion (probs BGR), also bit slower?
    return im


def plotter(im):
    im = pylab.array(im)
    
    max_val = np.max(im)
    print(max_val)
    
    if max_val <= 255:   # 8bit uint
        vmax = 255
    elif max_val <= 65536-1:  # 16bit uint
        vmax = 65536
    
    # im = im[::, 8000:12000:1, ... ]
    im = im[::, ::10, ... ]
    # im = np.mean(im, axis = 2)
    pylab.imshow(im/vmax, vmax = 1, interpolation='nearest', cmap='gray')
    pylab.show()


def foo_a(folder, name_list):
    sizes_tif = []
    for im_name_i in name_list:
        path_i = folder + im_name_i

        im = path2im(path_i)

        size = im.shape  # -1 don't change dtype
        
        sizes_tif.append(size)
            
        print('{} : {}'.format(im_name_i, size))

        plotter(im)
            

    meta_list = zip(name_list, sizes_tif)
    return meta_list


def good_images():
    # 3.5m high
    irr = '13IRRASS0001.tiff'    # res = (4973-19034)pixels / 1.106 m
    xr = '_8_panelXRR53.tif'    # res = (33691 - 9773) pixels / 1.106 m (and rotation!!)
    
    
    rgb = '13MCRIRX0010.tif'
    im = path2im( mypath + rgb)
    plotter(im)
    
    
mypath = '/scratch/lameeus/data/altarpiece_close_up/beard/'


def show_histo(array):
    intens, bins = np.histogram(np.reshape(array, (-1)), bins=256, range=[0, 1])
    
    bins_center = (bins[0:-1] + bins[1:]) / 2.0
    
    plt.plot(bins_center, intens)
    plt.show()


def generate_y():
    path = '/scratch/Dropbox/_PhD/THUIS WERKEN/Link to input/lam_anot.tif'

    im = path2im(path)
    array = np.array(im)
    
    # Throw away transparancy
    array = array[:, :, 0:3]
    
    # plt.imshow(array)
    # plt.show()
    
    print(np.shape(array))

    show_histo(array[:,:,:]/255)
    

    color = np.asarray([255, 0, 0]) # red

    a = np.equal(array[:,:, 0], color[0])
    b = np.equal(array[:,:, 1], color[1])
    c = np.equal(array[:,:, 2], color[2])
    foo = np.where(np.logical_and(np.logical_and(a, b), c))

    print(foo)
    
    bar = np.zeros(shape=np.shape(array))

    foo =  np.transpose(foo)
    
    for i in range(len(foo)):
        x = foo[i][0]
        y = foo[i][1]
        bar[x,y,:] = 1.
    
    
    plt.imshow(bar)
    plt.show()

def main():
    generate_y()
    
    # good_images()
    #
    #
    # from os import listdir
    # from os.path import isfile, join
    # onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    #
    # only_tif = [f_name for f_name in onlyfiles if ('.tif' == f_name[-4:] or '.tiff' == f_name[-5:] )]    #
    #
    # print(onlyfiles)
    # print(only_tif)
    #
    # n_start = 4
    # meta_list = time_func(lambda: foo_a(mypath, only_tif[n_start:]))
    #
    # print(meta_list)
    #
    # # import image
    #
    # # test_tif = only_tif[0]
    # # # img = image.load_img(mypath + test_tif, target_size = None)
    # # # x = image.img_to_array(img)
    # #
    # # plt.imshow(img)
    # # plt.figure()
    # # plt.imshow(x)
    # # plt.show()

if __name__ == '__main__':
    main()
