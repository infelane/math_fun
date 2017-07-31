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
import keras
import os, sys

import keras_ipi

import config4
import lambnet

folder_loc = '/ipi/private/lameeus/private_Documents/python/2017_February/super_res_challenge'
cmd_subfolder = os.path.realpath(folder_loc)
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import data_net

from paint_tools import image_tools


def time_func(func, n = 1):
    start = time.time()
    for i in range(n):
        a = func()
    end = time.time()
    print('elapsed time: {} s'.format(end - start))
    return a


def plotter(im):
    im = pylab.array(im)
    
    max_val = np.max(im)
    print(max_val)

    if max_val <= 1:    # Float.
        vmax = 1.
    elif max_val <= 255:   # 8bit uint
        vmax = 255.
    elif max_val <= 65536-1:  # 16bit uint
        vmax = 65536.
    
    # im = im[::, 8000:12000:1, ... ]
    im = im[::, ::10, ... ]
    # im = np.mean(im, axis = 2)
    pylab.imshow(im/vmax, vmax = 1, interpolation='nearest', cmap='gray')
    pylab.show()


def foo_a(folder, name_list):
    sizes_tif = []
    for im_name_i in name_list:
        path_i = folder + im_name_i

        im = image_tools.path2im(path_i)

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
    im = image_tools.path2im( mypath + rgb)
    plotter(im)
    
    
mypath = '/scratch/lameeus/data/altarpiece_close_up/beard/'


def show_histo(array):
    intens, bins = np.histogram(np.reshape(array, (-1)), bins=256, range=[0, 1])
    
    bins_center = (bins[0:-1] + bins[1:]) / 2.0
    
    plt.plot(bins_center, intens)
    plt.show()


def info_output(out):
    if out == 0:
        print("background")
    if out == 1:
        print("paint loss")
    if out == -1:
        print("no info")
        
        
def remove_transparancy(array):
    return array[:,:,0:3]


def map_redish(array):
    a = np.greater(array[:, :, 0], 0.7)
    b = np.greater(0.3, array[:, :, 1])
    c = np.greater(0.3, array[:, :, 2])
    
    return np.logical_and(np.logical_and(a, b), c)


def map_color(array, color):
    """
    return a binary map of where the color is present
    :param array:
    :param color:
    :return:
    """
    
    equals = []
    for i in range(3):
        equals.append(np.equal(array[:,:, i], color[...,  i]))
    
    # a = np.equal(array[:,:, 0], color[0])
    # b = np.equal(array[:,:, 1], color[1])
    # c = np.equal(array[:,:, 2], color[2])
    return np.logical_and(np.logical_and(equals[0], equals[1]), equals[2])


#TODO
def map2annot(map):
    ...
    

def get_input_all_big():
    folder = '/home/lameeus/data/ghent_altar/input/'
    im_clean = image_tools.path2im(folder + '13_clean.tif')
    im_rgb = image_tools.path2im(folder + '13_new_rgb_reg0.tif')
    im_ir = image_tools.path2im(folder + '13_new_ir_reg0.tif')

    im_all = np.concatenate([im_clean[:,:,0:3], im_rgb[:,:,0:3], np.mean(im_ir[:,:,0:3], axis = 2, keepdims=True)], axis = 2)
    return im_all


def get_input_all():
    folder = '/scratch/lameeus/data/ghentaltarpiece/altarpiece_close_up/beard_updated/'
    im_clean = image_tools.path2im(folder + 'rgb_cleaned.tif')
    im_rgb = image_tools.path2im(folder + 'rgb.tif')
    im_ir = image_tools.path2im(folder + 'ir_non_refl.tif')

    im_all = np.concatenate([im_clean[:,:,0:3], im_rgb[:,:,0:3], np.mean(im_ir[:,:,0:3], axis = 2, keepdims=True)], axis = 2)
    return im_all


def get_input(set):
    if set == 'big_v1': # good one
        folder = '/home/lameeus/data/ghent_altar/input/'
        im_clean = image_tools.path2im(folder + '13_new_clean_reg1.tif')[:, :, 0:3]
        im_rgb = image_tools.path2im(folder + '13_new_rgb_reg1.tif')[:, :, 0:3]
        im_ir = np.stack([image_tools.path2im(folder + '13_new_ir_reg1.tif')], axis = 2)
    elif set == 'hand':
        folder = '/home/lameeus/data/ghent_altar/input/'
        im_clean = image_tools.path2im(folder + '19_clean_crop_scale.tif')[:, :, 0:3]
        im_rgb = image_tools.path2im(folder + '19_rgb.tif')[:, :, 0:3]
        im_ir = np.stack([image_tools.path2im(folder + '19_ir_single.tif')], axis=2)
    
    else:
        raise NotImplementedError
    
    im_all = np.concatenate([im_clean, im_rgb, im_ir], axis=2)
    return im_all


def gen_y_part1_1():
    folder_foo = '/home/lameeus/data/ghent_altar/annotation/'
    annot_foo = folder_foo + 'data_2BD.tif'
    
    im_foo = image_tools.path2im(annot_foo)
    im_foo = remove_transparancy(im_foo)
    
    black = [0., 0., 0.]
    red = np.reshape([1., 0., 0.], (1,1,3))
    blue = [0., 0., 1.]
    
    map_red = map_color(im_foo, red)
    # map_red = map_redish(im_foo)
    
    im_bar = im_foo[...]
    im_bar[...] = np.reshape(blue, (1, 1, 3))
    im_bar[map_red] = red
    
    map_foo = color2map(im_bar)
    
    image_tools.save_im(im_bar, '/home/lameeus/data/ghent_altar/annotation/19_3.tif')
    
    plt.imshow(im_bar)
    plt.show()
    1/0
    
    
def gen_y_part1(set):
    # gen_y_part1_1()
    
    folder_big = '/home/lameeus/data/ghent_altar/input/'
    if set == 'zach':
        path_ground = folder_big + '13_clean.tif'
    elif set == 'hand':
        path_ground = folder_big + '19_clean_crop_scale.tif'
        
    im_ground = image_tools.path2im(path_ground)
    array_big = np.array(im_ground)
    
    red = np.reshape([1., 0., 0. ], (1,1,3))
    
    if set == 'zach':
        im_close_clean = image_tools.path2im('/scratch/lameeus/data/ghentaltarpiece/altarpiece_close_up/beard_updated/rgb_cleaned.tif')
        im_close_clean = remove_transparancy(im_close_clean)


        # path = '/home/lameeus/data/ghent_altar/annotation/lam_anot.tif'
        
        path = '/home/lameeus/data/ghent_altar/annotation/13_shaoguang.tif'
    
        im = image_tools.path2im(path)
        array = np.array(im)
        shape = np.shape(array)
        # Throw away transparancy
        array = remove_transparancy(array)
    

        map_red = map_color(array, red)
    
        im_close_clean[map_red] = red



    
    folder_annot = '/home/lameeus/data/ghent_altar/annotation/'
    if set == 'zach':
        # Zachary,
        # P1: 790, 790
        # P2: 371, 150
        # P3: 912, 1623​
        
        im_annot1 = remove_transparancy(image_tools.path2im(folder_annot + '13_1.tif')) #'data_1BDcorrected.tif'
        h1 = 1698 - 406
        w1 = 4059 - 3557
        shape1 = np.shape(im_annot1)
        im_close_clean[h1:h1 + shape1[0], w1 : w1 + shape1[1], :] = im_annot1
        
        im_annot2 = remove_transparancy(image_tools.path2im(folder_annot + '13_2.tif')) #'Zachary_P2FINAL.tif'
        h1 = 150
        w1 = 371
        shape1 = np.shape(im_annot2)
        im_close_clean[h1:h1 + shape1[0], w1 : w1 + shape1[1], :] = im_annot2
        
        im_annot3 = remove_transparancy(image_tools.path2im(folder_annot + '13_3.tif')) # 'Zachary_P3FINAL.tif'
        h1 = 1623
        w1 = 912
        shape1 = np.shape(im_annot3)
        im_close_clean[h1:h1 + shape1[0], w1 : w1 + shape1[1], :] = im_annot3
        
        
        # fill small piece in big
        h_start_close = 406
        w_start_close = 3557
        array_big[h_start_close:h_start_close + shape[0], w_start_close:w_start_close + shape[1], :] = im_close_clean
    
    elif set == 'hand':
        im_annot1 = remove_transparancy(image_tools.path2im(folder_annot + '19_1.tif'))  #'John_the_Evangelist_P1FINAL.tif'
        im_annot2 = remove_transparancy(image_tools.path2im(folder_annot +  '19_2.tif' ))   # 'John_the_Evangelist_P2FINAL.tif'
        im_annot3 = remove_transparancy(image_tools.path2im(folder_annot +  '19_3.tif' )) #'data_2BD.tif'

        h1 = 1993
        w1 = 1494
        rescale = 0.3653006382
        foo = cv2.resize(im_annot1, (0, 0), fx=rescale, fy=rescale, interpolation= 0)
        shape1 = np.shape(foo)
        array_big[h1:h1 + shape1[0], w1: w1 + shape1[1], :] = foo[...]

        h1 = 1721
        w1 = 1314
        rescale = 0.3653006382
        foo = cv2.resize(im_annot2, (0, 0), fx=rescale, fy=rescale, interpolation=0)
        shape1 = np.shape(foo)
        array_big[h1:h1 + shape1[0], w1: w1 + shape1[1], :] = foo[...]

        h1 = 1608
        w1 = 963
        rescale = 0.3653006382
        foo = cv2.resize(im_annot3, (0, 0), fx=rescale, fy=rescale, interpolation=0)
        shape1 = np.shape(foo)
        array_big[h1:h1 + shape1[0], w1: w1 + shape1[1], :] =foo[...]

    
    # # Second annotation
    # path = '/home/lameeus/data/ghent_altar/annotation/corr1.tif'
    # im = image_tools.path2im(path)
    # array = np.array(im)
    # array = remove_transparancy(array)
    # shape = np.shape(array)
    #
    # # fill small piece in big
    # h_start_bart = 1698  # 400
    # w_start_bart = 4059  # 3560
    # array_big[h_start_bart:h_start_bart + shape[0], w_start_bart:w_start_bart + shape[1], :] = array
    

    # plt.imshow(array_big[:,:, :], interpolation = 'nearest')   #[1500:1900, 4600:5000, :]
    # plt.show()
    
    #
    # print(np.shape(array))
    #
    # # show_histo(array[:,:,:])
    #
    #
    #
    # coords_info = np.where(red_map)
    #
    # # import scipy.misc
    # # im_save = im_ground[h_start_bart:h_start_bart + shape[0], w_start_bart:w_start_bart + shape[1], :]
    # # im_save[red_map] = red
    # # im_save[blue_map] = blue
    # #
    # # scipy.misc.imsave('test.tif', im_save)
    #
    #
    #
    #
    #
    # # plt.imshow(class_map)
    # # plt.show()
    #
    # print(zip(coords_info))
    #
    annot_big_color = np.zeros(shape=np.shape(array_big))
    
    # plt.imshow(array_big, vmax = 1.)
    # plt.show()
    

    class_map = color2map(array_big)

    blue = [0., 0., 1.]
    #
    annot_big_color[class_map == 1] = np.reshape(red, newshape=(1, 1, -1))
    annot_big_color[class_map == 0] = np.reshape(blue, newshape=(1, 1, -1))
    annot_big_color[class_map == -1] = np.reshape(np.array([1, 1, 1]), newshape=(1, 1, -1))

    plt.imshow(annot_big_color, vmax = 1.)
    plt.show()
    
    folder = '/home/lameeus/data/ghent_altar/input/'
    
    if set == 'zach':
        image_tools.save_im(annot_big_color, folder + '13_annot.tif')
    elif set == 'hand':
        image_tools.save_im(annot_big_color, folder + '19_annot.tif')

    annot_to_annotclean(annot_big_color, set)
    
    
def gen_y_part2():
    """
    Build the new big images
    :return:
    """
    
    im_all_small = get_input_all()
    im_all = get_input_all_big()

    clean = im_all[:,:,0:3]
    rgb = im_all[:,:,3:6]
    ir = im_all[:,:,6]
    clean_small = im_all_small[:,:,0:3]
    rgb_small = im_all_small[:,:,3:6]
    ir_small = im_all_small[:,:,6]

    shape = np.shape(clean_small)

    h_start_close = 406
    w_start_close = 3557

    clean[h_start_close:h_start_close+shape[0], w_start_close:w_start_close+shape[1]] = clean_small
    rgb[h_start_close:h_start_close+shape[0], w_start_close:w_start_close+shape[1]] = rgb_small
    ir[h_start_close:h_start_close+shape[0], w_start_close:w_start_close+shape[1]] = ir_small

    folder = '/home/lameeus/data/ghent_altar/input/'
    # save new image
    scipy.misc.toimage(clean, cmin=0.0, cmax=1.0).save(folder + '13_new_clean_reg1.tif')
    scipy.misc.toimage(rgb, cmin=0.0, cmax=1.0).save(folder + '13_new_rgb_reg1.tif')
    scipy.misc.toimage(ir, cmin=0.0, cmax=1.0).save(folder + '13_new_ir_reg1.tif')
    
    plt.subplot(2,2,1)
    plt.imshow(clean)
    plt.subplot(2,2,2)
    plt.imshow(rgb)
    plt.subplot(2,2,3)
    plt.imshow(ir)
    plt.show()


def annot_to_annotclean(annot, set):
    # if set == 'zach':
    #     im_all = get_input_all_big()
    # elif set == 'hand':
    im_all = get_input(set)
    clean = remove_transparancy(im_all)
    
    red = np.asarray([1., 0., 0.])  # red, paint loss
    blue = np.asarray([0., 0., 1.])  # blue, background

    red_map = map_color(annot, red)
    blue_map = map_color(annot, blue)
    
    clean[red_map] = red
    clean[blue_map] = blue
    
    if set == 'zach':
        path = '/home/lameeus/data/ghent_altar/input/' + '13_annot_clean.tif'
    elif set == 'hand':
        path = '/home/lameeus/data/ghent_altar/input/' + '19_annot_clean.tif'
    image_tools.save_im(clean, path)


def color2map(image):
    red = np.asarray([1., 0., 0.])  # red, paint loss
    blue = np.asarray([0., 0., 1.])  # blue, background
    
    red_map = map_color(image, red)
    # red_map = map_redish(array) #TODO
    
    blue_map = map_color(image, blue)
    
    class_map = np.empty(shape=np.shape(image)[0:2], dtype=np.int8)
    class_map[...] = -1  # No info
    class_map[red_map] = 1  # paint loss
    class_map[blue_map] = 0  # background

    return class_map


def generate_y(ext, set):
    gen_y_part1(set)
    # gen_y_part2()

    folder = '/home/lameeus/data/ghent_altar/input/'

    if set == 'hand':
        im_all = get_input('hand')
        annot_big_color = image_tools.path2im(folder + '19_annot_clean.tif')
    if set == 'zach':
        im_all = get_input('big_v1')
        annot_big_color = image_tools.path2im(folder + '13_annot_clean.tif')

    class_map = color2map(annot_big_color)

    coords_info = np.where(np.logical_or(class_map == 1, class_map == 0))
    print(coords_info)
    
    annot_big_color[coords_info] = np.reshape(np.array([0,0,0])/255, newshape=(1, 1, -1))
    
    x_inputs = []
    y_outputs = []

    width = 3    # segment more than one pixel at once
    im_seg = data_net.SegmentedImage(im_all, ext = ext + width)
    class_map_seg = data_net.SegmentedImage(class_map, ext = width)
    
    idx = np.arange(len(coords_info[0]))
    np.random.seed(1)
    np.random.shuffle(idx)
    
    n_pixels = min(100000, len(coords_info[0]))
    n_pixels = len(coords_info[0])
    
    print('amount of patches: {}'.format(n_pixels))
    
    for i_pixel in range(n_pixels):
        # shuffling
        x_co = coords_info[0][idx[i_pixel]]
        y_co = coords_info[1][idx[i_pixel]]

        input_patch = im_seg.get_patch(x_co, y_co, orig_width=1)
        
        
        # plt.subplot(2,2,1)
        # plt.imshow(input_patch[:,:,0:3])
        # plt.subplot(2, 2, 2)
        # plt.imshow(input_patch[:, :, 3:6])
        # plt.subplot(2, 2, 3)
        # plt.imshow(input_patch[:, :, 6], cmap = 'Greys')
        # plt.show()
        
        # input_patch = im_all[x_co-ext:x_co+ext+1, y_co-ext:y_co+ext+1,:]
        output_patch = np.zeros(shape=(1 + 2*width, 1 + 2*width, 2), dtype=float)
    
        output_patch_map = class_map_seg.get_patch(x_co, y_co, orig_width=1)

        # output_patch[np.equal(output_patch_map, 0), 0] = 1.

        # print(output_patch)

        for i_class in range(2):
            uno = np.equal(output_patch_map, i_class).astype(float)
            output_patch[:,:,i_class] = uno
        
        # print(output_patch)
        
        # if class_map[x_co, y_co] == 0:
        #     output_patch[..., 0] = 1.
        # elif class_map[x_co, y_co] == 1:
        #     output_patch[..., 1] = 1.

        x_inputs.append(input_patch)
        y_outputs.append(output_patch)
        
    def save_it(x_inputs, y_outputs, set):
        x_inputs = np.stack(x_inputs, axis=0)
        y_outputs = np.stack(y_outputs, axis=0)
    
        if set == 'zach':
            np.savez_compressed('xy_comp_ext7_100000.npz', x=x_inputs, y=y_outputs)
        elif set == 'hand':
            np.savez_compressed('xy_hand_ext7.npz', x=x_inputs, y=y_outputs)

    if 1:
        save_it(x_inputs, y_outputs, set)
    
    # for i in range(100):
    #     plt.imshow(x_inputs[i, :, :, 0:3])
    #     plt.title(y_outputs[i, :, :, 1])
    #     plt.show()
    
    

    # plt.figure()
    # plt.imshow(annot_big_color)
    # plt.show()
    
    
def registration():
    folder = '/home/lameeus/data/ghent_altar/input/'
    ir = image_tools.path2im(folder+'13_ir.tif')
    rgb =image_tools.path2im(folder+'13_rgb.tif')
    clean =image_tools.path2im(folder+'13_clean.tif')
    
    # x than y
    # # left point of banner
    # coords_1_ir = np.array([646, 2269])     # correct
    # coords_1_rgb = np.array([631, 2138])    # correct
    # coords_1_clean = np.array([1042, 3193 ])    # correct
    # left crack + letter
    coords_1_ir = np.array([584, 1951])
    coords_1_rgb = np.array([577, 1833])
    coords_1_clean = np.array([968, 2812])
    
    # Right point of banner
    coords_2_ir = np.array([5501, 1094 ])   # correct
    coords_2_rgb = np.array([5313, 1098 ])  # correct
    coords_2_clean = np.array([6935, 1837 ])    # correct
    
    diff_clean = coords_2_clean- coords_1_clean
    diff_rgb = coords_2_rgb- coords_1_rgb
    diff_ir = coords_2_ir- coords_1_ir
    
    dist_clean = np.sqrt(np.sum(np.square(diff_clean)))
    dist_rgb = np.sqrt(np.sum(np.square(diff_rgb)))
    dist_ir = np.sqrt(np.sum(np.square(diff_ir)))
    
    angle_clean = np.arctan2(diff_clean[1], diff_clean[0])
    angle_rgb = np.arctan2(diff_rgb[1], diff_rgb[0])
    angle_ir = np.arctan2(diff_ir[1], diff_ir[0])
    # print(angle_clean)
    # print(angle_rgb)
    # print(angle_ir)
    
    dif_angle_rgb =  (angle_clean-angle_rgb)*180/np.pi
    dif_angle_ir = (angle_clean-angle_ir)*180/np.pi
    
    # print(dif_angle_rgb)
    # print(dif_angle_ir)
    
    rescale_rgb = dist_clean/dist_rgb
    rescale_ir = dist_clean/dist_ir

    rgb_center = np.array(np.shape(rgb)[1::-1])/2.
    clean_center = np.array(np.shape(clean)[1::-1])/2.
    ir_center = np.array(np.shape(ir)[1::-1])/2.
    
    vector_clean_1 = coords_1_clean - clean_center
    vector_clean_2 = coords_2_clean - clean_center
    
    vector_rgb_1 = coords_1_rgb - rgb_center
    vector_rgb_2 = coords_2_rgb - rgb_center

    vector_ir_1 = coords_1_ir - ir_center
    vector_ir_2 = coords_2_ir - ir_center
    
    new_rgb = np.empty(shape=np.shape(clean))
    new_rgb[:,:,:] = 0.5
    new_ir = np.copy(new_rgb[:,:,:])
    
    # import PIL
    # im_rgb = Image.frombytes(rgb)
    # im_rgb = im_rgb.rotate(dif_angle_rgb)
    # rgb.rotate()

    def rot_vector(vector, angle):
        """
        
        :param vector:
        :param angle_rad:  in degree
        :return:
        """
        angle_rad = angle* np.pi / 180
        vector_out = np.zeros(np.shape(vector))
        vector_out[0] = np.cos(angle_rad) * vector[0] - np.sin(angle_rad) * vector[1]
        vector_out[1] = np.sin(angle_rad) * vector[0] + np.cos(angle_rad) * vector[1]
        
        return vector_out
    
    # TODO IS x and y axis correct?? Angle oposite
    rgb = scipy.ndimage.interpolation.rotate(rgb, -dif_angle_rgb, cval = 0.5, output=float)
    ir = scipy.ndimage.interpolation.rotate(ir, -dif_angle_ir, cval = 0.5, output=float)
    
    # print(vector_rgb_1)   # TODO ANGLES SHOULD BE SUBSTRACTED
    vector_rgb_1 = rot_vector(vector_rgb_1, dif_angle_rgb)
    vector_rgb_2 = rot_vector(vector_rgb_2, dif_angle_rgb)
    vector_ir_1 = rot_vector(vector_ir_1, dif_angle_ir)
    vector_ir_2 = rot_vector(vector_ir_2, dif_angle_ir)


    rgb = cv2.resize(rgb,(0, 0), fx = rescale_rgb, fy = rescale_rgb )
    ir = cv2.resize(ir, (0, 0), fx = rescale_ir, fy = rescale_ir )
    
    # rgb = scipy.misc.imresize(rgb, rescale_rgb)/255
    # ir = scipy.misc.imresize(ir, rescale_ir)/255 # colors are changed, not good!
    
    vector_rgb_1 = vector_rgb_1*rescale_rgb
    vector_rgb_2 = vector_rgb_2*rescale_rgb
    vector_ir_1 =  vector_ir_1*rescale_ir
    vector_ir_2 = vector_ir_2*rescale_ir
    
    print(vector_clean_1 - vector_rgb_1)
    print(vector_clean_2 - vector_rgb_2)
    print(vector_clean_1 - vector_ir_1)
    print(vector_clean_2 - vector_ir_2)
    
    rgb_center_new = np.array(np.shape(rgb)[1::-1])/2.
    ir_center_new = np.array(np.shape(ir)[1::-1])/2.
    
    print(clean_center - rgb_center_new)
    print(((vector_clean_1 - vector_rgb_1) + (vector_clean_2 - vector_rgb_2))/2)
    
    move_rgb = (clean_center - rgb_center_new) + ((vector_clean_1 - vector_rgb_1) + (vector_clean_2 - vector_rgb_2))/2
    move_ir = clean_center - ir_center_new + ((vector_clean_1 - vector_ir_1) + (vector_clean_2 - vector_ir_2))//2
    
    plt.subplot(2, 2, 1)
    plt.imshow(clean)
    plt.title('clean')
    
    plt.subplot(2, 2, 2)
    shape_rgb = np.shape(rgb)
    h_start = int(move_rgb[1])      # - 400-20   +1337-1339 +1337-1255
    h_end = h_start + shape_rgb[0]
    w_start = int(move_rgb[0])      # -5   +1819-1840   +1819-1918
    w_end = w_start + shape_rgb[1]
    print(shape_rgb)
    print(h_start)
    print(h_end)
    print(w_start)
    print(w_end)
    new_rgb[h_start:h_end, w_start:w_end, :] = rgb
    plt.imshow(new_rgb)
    # plt.imshow(rgb)
    plt.title('rgb')
    
    plt.subplot(2, 2, 3)
    shape_ir = np.shape(ir)
    h_start = int(move_ir[1]) #-311
    h_end = h_start + shape_ir[0]
    w_start = int(move_ir[0]) #- 75
    w_end = w_start + shape_ir[1]
    print(shape_ir)
    print(h_start)
    print(h_end)
    print(w_start)
    print(w_end)
    new_ir[h_start:h_end, w_start:w_end, :] = ir
    plt.imshow(new_ir)
    plt.title('ir')
    
    plt.subplot(2, 2, 4)
    ir_mean = np.mean(new_ir, axis=2)
    rgb_mean = np.mean(new_rgb, axis=2)
    clean_mean = np.mean(clean, axis=2)
    combo = np.stack([clean_mean, ir_mean, rgb_mean], axis = 2)
    plt.imshow(combo)
    plt.title('combo')
    plt.show()

    scipy.misc.toimage(new_ir, cmin=0.0, cmax=1.0).save(folder + '13_new_ir_reg0.tif')
    scipy.misc.toimage(new_rgb, cmin=0.0, cmax=1.0).save(folder + '13_new_rgb_reg0.tif')
    
    
def load_xy(set ):
    
    if set == 'combo':
        xy = np.load('xy_comp_ext7_100000.npz')
        xy2 = np.load('xy_hand_ext7.npz')
        # combine both
    
        x = np.concatenate([xy['x'], xy2['x']], axis=0)
        y = np.concatenate([xy['y'], xy2['y']], axis=0)
    
        return x, y
    
    if set == 'zach':
        xy = np.load('xy_comp_ext7_100000.npz')
        
    elif set == 'hand':
        xy = np.load('xy_hand_ext7.npz')
    elif set == 'idk':
        xy = np.load('xy_comp.npz')
    
    return xy['x'], xy['y']


def loader(width, ext):
    layers = config4.nn3(width=width, ext=ext)
    # layers = config4.nn2(width=width, ext=ext)
    
    # if set == 'hand':
    #     # filepath = 'net_weights_hand_7.h5'
    #     filepath = 'net_weights_hand_7_v2.h5'
    # elif set == 'zach':
    #     filepath = 'net_weights_7.h5'
    
    # filepath = 'net_weights_hand_7.h5'
    # filepath = 'net_weights_hand_7_v2.h5'
    folder_weight = '/home/lameeus/data/ghent_altar/net_weight/'
    filepath = folder_weight + 'net_weights_4.h5'
    
    return layers, filepath


def train_net(ext, set):
    # import pickle
    # batch_train = pickle.load(open("/ipi/private/lameeus/private_Documents/python/2017_05/batch_train.p", "rb"))
    # n_subset = None
    # x = batch_train.x[:n_subset]
    # y = batch_train.y[:n_subset]
    # ext = 7
    
    x, y = load_xy('combo')  # width = 1, ext = ext
    
    width = 7

    balance = np.sum(y, axis = (0, 1, 2))
    print('balance: {} - {}'.format(balance[0], balance[1]))

    layers, filepath = loader(width, ext)
    model = keras_ipi.block_builder.stack(layers)
    
    print(model.summary())
    
    flag = config4.flag()

    metrics = []
    
    # loss = keras_ipi.losses.test_cost()
    loss = keras_ipi.losses.weigthed_crossentropy([1, 1], normalize = True)
    loss = keras_ipi.losses.bin_square()
    # loss = keras.losses.mean_squared_error
    optimizer = {'class_name': 'adam', 'config': {'lr': flag.lr, 'beta_1': 0.90}} #otherwise  = 'adam'

    model.compile(loss = loss,
                  optimizer=optimizer,
                  metrics=metrics
                  )

    if flag.bool_prev:
        depth = len(model.layers)

        # filepath_test = 'net_weights_3.h5'
        model.load_weights(filepath, depth=depth)

    checkpoint = keras.callbacks.ModelCheckpoint(filepath, verbose=0,
                                    save_weights_only=True, period=1)
    callbacks_list = [checkpoint]
    
    model.fit(x, y,
              batch_size=flag.batch_size, epochs=flag.epochs, shuffle=True,
              verbose=1,  # how much information to show 1 much or 0, nothing
              # class_weight= (1.0, 10.0),
              # validation_data=(X_test[:10000], Y_test[:10000]),
              callbacks=callbacks_list
              )
    

def test_net(ext, set):
    width = 8
    ext = ext

    layers, filepath = loader(width, ext)
    model = keras_ipi.block_builder.stack(layers)
    
    depth = len(model.layers)
    model.load_weights(filepath, depth=depth)

    info = lambnet.block_info.Info(model)
    # info.output_test(width, ext, set='zach_small')
    # info.output_vis(width, ext, set = 'zach_small', bool_save = False, last_layer = False)
    info.certainty(width, ext, set = 'zach')
    # info.certainty(width, ext, set = 'zach_small')
    # info.certainty(width, ext, set = 'hand')


def main():
    # registration()
    ext = 7
    set = 'zach'
    # generate_y(ext, set = set)
    train_net(ext, set)
    test_net(ext, set)
    
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
