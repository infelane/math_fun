""" Building block for data """

#3th party
from PIL import Image
import numpy as np
import sys, os
import pickle

#own
folder_loc = '/ipi/private/lameeus/private_Documents/python/2017_February/super_res_challenge'
cmd_subfolder = os.path.realpath(folder_loc)
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
    
import data_net
import block_pre

# global settings
save_path = '/scratch/lameeus/data/lamb/xy/'
bool_new_data = False

# raw data example
def raw_data(ex_raw_data, width, ext): #(ex_raw_data_i):
    # opening images

    def detectRed(im):
        return np.logical_and(im[..., 0] > 0.7, im[..., 1] < 0.2).astype(float)

    im_in_1 = ex_raw_data.im_in_1
    im_in_2 = ex_raw_data.im_in_2
    im_in_3 = ex_raw_data.im_in_3
    im_out = ex_raw_data.im_out

    # stacking in and output
    shape = np.shape(im_out)
    im_label1 = np.zeros(shape=list(shape[0 : 2]) + [2])
    
    im_in1 = np.concatenate((im_in_1, im_in_2, im_in_3[..., 0:1]), axis=-1)
    im_label1[..., 1] = detectRed(im_out)
    im_label1[..., 0] = 1 -  im_label1[..., 1]

    # convert to Data class
    data = data_net.Data(im_in1, im_label1, colors_sep=False, big_patches=width, ext = ext, bool_tophat = False)

    return data


def im2array(path):
    return np.asarray(Image.open(path)) / 255

def preproc(a):
    lab = block_pre.rgb2lab(a)
    
    # lab has range (0, 100), (-100, 100), (-100, 100)
    # Lets say it is even distribution:
    # mean = (b + a)/2
    # std = (b - a)/(2*sqrt(3))
    return (lab - (50.0, 0.0, 0.0)) / (28.87, 57.74 , 57.74)

def preproc_inv(a):
    b = a * (28.87, 57.74 , 57.74) + (50.0, 0.0, 0.0)
    return block_pre.lab2rgb(b)


class ExRawImage(object):
    def __init__(self, input_1, input_2, input_3, out, preprocessing = preproc):
        # preprocessing is only on the input RBG images
            
        self.im_in_1 = im2array(input_1)
        self.im_in_2 = im2array(input_2)
        self.im_in_3 = im2array(input_3)
        self.im_out = im2array(out)

        # # TODO resize
        # def resize(a):
        #
        # # from PIL import Image
        #
        #     import scipy
        #
        #     print(np.max(a))
        #
        #     b = scipy.misc.imresize(a, 1.0, interp = 'bicubic')/255.0
        #
        #     print(np.max(b))
        # #
        #     return b
        # #
        # self.im_in_1 = resize(self.im_in_1)
        # self.im_in_2 = resize(self.im_in_2)
        # self.im_in_3 = resize(self.im_in_3)
        # self.im_out = resize(self.im_out)
        
        if preprocessing:
            op = lambda a: preprocessing(a)
            self.im_in_1 = op(self.im_in_1)
            self.im_in_2 = op(self.im_in_2)
            self.preproc = True
        else:
            self.preproc = False
                
    def set_crop(self, h0, h1, w0, w1):
        self.im_in_1 = self.im_in_1[h0:h1, w0:w1, ...]
        self.im_in_2 = self.im_in_2[h0:h1, w0:w1, ...]
        self.im_in_3 = self.im_in_3[h0:h1, w0:w1, ...]
        
    def rotate(self, k):
        self.im_in_1 = np.rot90(self.im_in_1, k, (0, 1))
        self.im_in_2 = np.rot90(self.im_in_2, k, (0, 1))
        self.im_in_3 = np.rot90(self.im_in_3, k, (0, 1))
        self.im_out = np.rot90(self.im_out, k, (0, 1))
        
    def mir(self):
        self.im_in_1 = np.fliplr(self.im_in_1)
        self.im_in_2 = np.fliplr(self.im_in_2)
        self.im_in_3 = np.fliplr(self.im_in_3)
        self.im_out = np.fliplr(self.im_out)
        
    def get_image(self):
        if self.preproc:
            return preproc_inv(self.im_in_1)
        else:
            return self.im_in_1


def ex_raw_zach():
    input_1 = "/scratch/lameeus/data/altarpiece_close_up/beard_updated/rgb_cleaned.tif"
    input_2 = "/scratch/lameeus/data/altarpiece_close_up/beard_updated/rgb.tif"
    input_3 = "/scratch/lameeus/data/altarpiece_close_up/beard_updated/ir_non_refl.tif"
    out = "/scratch/lameeus/data/altarpiece_close_up/beard_updated/ground_truth.tif"

    return ExRawImage(input_1, input_2, input_3, out)


def ex_raw_zach_close():
    input_1 = "/scratch/lameeus/data/altarpiece_close_up/beard_updated/rgb_cleaned.tif"
    input_2 = "/scratch/lameeus/data/altarpiece_close_up/beard_updated/rgb.tif"
    input_3 = "/scratch/lameeus/data/altarpiece_close_up/beard_updated/ir_non_refl.tif"
    out = '/scratch/lameeus/data/altarpiece_close_up/ground_truth/data_1BDcorrected.tif'
    
    ex_raw_data = ExRawImage(input_1, input_2, input_3, out)
    
    h_start = 1292
    w_start = 502
    h_width = 400
    w_width = 400
    ex_raw_data.set_crop(h_start, h_start + h_width, w_start, w_start + w_width)
    return ex_raw_data


def ex_raw_hand():
    input_1 = "/scratch/lameeus/data/altarpiece_close_up/finger/hand_cleaned.tif"
    input_2 = "/scratch/lameeus/data/altarpiece_close_up/finger/hand_rgb.tif"
    input_3 = "/scratch/lameeus/data/altarpiece_close_up/finger/hand_ir.tif"
    output_1 = "/scratch/lameeus/data/altarpiece_close_up/finger/ground_truth.tif"

    return ExRawImage(input_1, input_2, input_3, output_1)


def ex_raw_hand_close():
    input_1 = "/scratch/lameeus/data/altarpiece_close_up/finger/hand_cleaned.tif"
    input_2 = "/scratch/lameeus/data/altarpiece_close_up/finger/hand_rgb.tif"
    input_3 = "/scratch/lameeus/data/altarpiece_close_up/finger/hand_ir.tif"
    out = '/scratch/lameeus/data/altarpiece_close_up/ground_truth/data_2BD.tif'

    ex_raw_data = ExRawImage(input_1, input_2, input_3, out)

    h_start = 200 - (355 - 181)
    w_start = 200 + 88

    h_width = 382
    w_width = 570
    ex_raw_data.set_crop(h_start, h_start + h_width, w_start, w_start + w_width)
    return ex_raw_data


def raw2data_gen(raw_list, width, ext):
    """
    Return the training data generator
    :param width:
    :param ext:
    :return:
    """
    
    in_patches = []
    out_patches = []
    for ex_raw_data_i in raw_list:
        data_i = raw_data(ex_raw_data_i, width, ext)
        in_patches.append(data_i.in_patches())
        out_patches.append(data_i.out_patches())
    
    patches_input = np.concatenate(in_patches, axis=0)
    patches_output = np.concatenate(out_patches, axis=0)
    
    return data_net.DataGen(images=patches_input, labels=patches_output)

    
def data_train(width, ext):

    list_ex = [ex_raw_zach, ex_raw_hand_close, ex_raw_zach_close]

    # raw_list = [ex_raw_zach(), ex_raw_hand_close(), ex_raw_zach_close()]

    import time

    # todo augmentation on each element

    start = time.time()
    
    raw_list = []
    for ex_i in range(3):
        for mir_i in range(2):
            for rot_i in range(4):
                name = "data_train_{}_{}_{}.p".format(ex_i, mir_i, rot_i)
                if bool_new_data:
                
                    raw_list_i = list_ex[ex_i]()
                    raw_list_i.rotate(rot_i)
                    if mir_i:
                        raw_list_i.mir()

                    pickle.dump(raw_list_i, open(save_path + name, "wb"))
                else:
                    raw_list_i = pickle.load(open(save_path + name, "rb"))
                        
                raw_list.append(raw_list_i)
                
                

    end = time.time()
    print('time1: {}'.format(end - start))
    
    start = time.time()

    a = raw2data_gen(raw_list, width, ext)

    end = time.time()
    print('time1: {}'.format(end - start))
    
    # pickle.dump(a, open(save_path + name, "wb"))
    return a


def data_valid(width, ext):
    name = "data_valid.p"

    if bool_new_data:
    
        raw_list = [ex_raw_hand()]
        
        a = raw2data_gen(raw_list, width, ext)

        pickle.dump(a, open(save_path + name, "wb"))
        return a
    else:
        return pickle.load( open(save_path + name, "rb" ) )


def test_data(set, width, ext, bool_new_data):
    """
    Hand or Zachary
    :return: the data and image after cleaning
    """
    if set == 'zach':
        ex_raw_i = ex_raw_zach
        name = "data_test_zach.p"
    
    elif set == 'hand':
        ex_raw_i = ex_raw_hand
        name = "data_test_hand.p"
        
    if bool_new_data:
    
        foo = ex_raw_i()
        

        

        # # todo
        # foo.rotate(2)
        
        ab = (raw_data(foo, width, ext), foo.get_image())
        
        pickle.dump(ab, open(save_path + name, "wb"))

        return ab
        
    else:
        return pickle.load(open(save_path + name, "rb"))
