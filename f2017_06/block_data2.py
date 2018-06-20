import numpy as np
from PIL import Image

from f2017_02.super_res_challenge import data_net
from link_to_soliton.paint_tools import image_tools

# def im2array(path):
#
#     foo =  np.asarray(Image.open(path)) #/ 255
#     foo =
#
#     return foo

class ExRawImage(object):
    def __init__(self, input_1, input_2, input_3, out):
        # preprocessing is only on the input RBG images
        
        self.im_in_1 = image_tools.path2im(input_1)
        self.im_in_2 = image_tools.path2im(input_2)
        self.im_in_3 = image_tools.path2im(input_3)
        self.im_out = image_tools.path2im(out)
        
    # def set_crop(self, h0, h1, w0, w1):
    #     self.im_in_1 = self.im_in_1[h0:h1, w0:w1, ...]
    #     self.im_in_2 = self.im_in_2[h0:h1, w0:w1, ...]
    #     self.im_in_3 = self.im_in_3[h0:h1, w0:w1, ...]
    #
    # def rotate(self, k):
    #     self.im_in_1 = np.rot90(self.im_in_1, k, (0, 1))
    #     self.im_in_2 = np.rot90(self.im_in_2, k, (0, 1))
    #     self.im_in_3 = np.rot90(self.im_in_3, k, (0, 1))
    #     self.im_out = np.rot90(self.im_out, k, (0, 1))
    #
    # def mir(self):
    #     self.im_in_1 = np.fliplr(self.im_in_1)
    #     self.im_in_2 = np.fliplr(self.im_in_2)
    #     self.im_in_3 = np.fliplr(self.im_in_3)
    #     self.im_out = np.fliplr(self.im_out)
    #
    # def get_image(self):
    #     if self.preproc:
    #         return preproc_inv(self.im_in_1)
    #     else:
    #         return self.im_in_1
    

def ex_raw_zach_small():
    folder = '/scratch/lameeus/data/ghent_altar/OLD/altarpiece_close_up/beard_updated/'
    input_1 = folder + "rgb_cleaned.tif"
    input_2 = folder + "rgb.tif"
    input_3 = folder + "ir.tif"
    folder_annot = '/scratch/lameeus/data/ghent_altar/annotation/'
    out = folder_annot + "13_zach_small_annot2.tif"#"ground_truth.tif"
    
    return ExRawImage(input_1, input_2, input_3, out)
    
    
def ex_raw_hand_small():
    if 0:
        folder = '/scratch/lameeus/data/ghent_altar/OLD/altarpiece_close_up/finger/'
        input_1 = folder + "hand_cleaned.tif"
        input_2 = folder + "hand_rgb.tif"
        input_3 = folder + "hand_ir.tif"
        out = folder + "ground_truth.tif"

    else:
        folder = '/scratch/lameeus/data/ghent_altar/input/19_hand/'
        input_1 = folder + "19_hand_clean.tif"
        input_2 = folder + "19_hand_rgb.tif"
        input_3 = folder + "19_hand_ir.tif"
        out = folder + "19_hand_annot.tif"

    return ExRawImage(input_1, input_2, input_3, out)


def ex_raw_hand_big():
    folder = '/scratch/lameeus/data/ghent_altar/input/'
    input_1 = folder + "19_clean.tif"
    out = folder + "19_annot_big.tif"

    folder = '/scratch/lameeus/data/ghent_altar/input/registration/'
    input_2 = folder + "19_rgb_reg.tif"
    input_3 = folder + "19_ir_reg.tif"

    return ExRawImage(input_1, input_2, input_3, out)


def ex_raw_zach():

    folder = '/scratch/lameeus/data/ghent_altar/input/'
    input_1 = folder + "13_new_clean_reg1.tif"
    input_2 = folder + "13_new_rgb_reg1.tif"
    input_3 = folder + "13_new_ir_reg1.tif"
    out = folder + "13_annot.tif"

    return ExRawImage(input_1, input_2, input_3, out)


def ex_raw_hand():
    folder = '/scratch/lameeus/data/ghent_altar/input/'
    input_1 = folder + "19_clean_crop_scale.tif"
    input_2 = folder + "19_rgb.tif"
    input_3 = folder + "19_ir_single.tif"
    # out = folder + "19_clean_crop_scale.tif"    # TODO this is not annotated
    out = folder + "19_annot.tif"
    
    return ExRawImage(input_1, input_2, input_3, out)
    

def raw_data(ex_raw_data, width, ext):  # (ex_raw_data_i):
    # opening images

    def detectRed(im):
        return np.logical_and(im[..., 0] > 0.7, im[..., 1] < 0.2).astype(float)

    im_in_1 = ex_raw_data.im_in_1
    im_in_2 = ex_raw_data.im_in_2
    im_in_3 = ex_raw_data.im_in_3
    
    im_out = ex_raw_data.im_out
    
    # # stacking in and output
    shape = np.shape(im_out)
    im_label1 = np.zeros(shape=list(shape[0: 2]) + [2])
    #
   
    # print(np.shape(im_in_3))
    
    # im_in1 = np.concatenate((im_in_1, im_in_2, np.mean(im_in_3[:,:, 0:3], axis = 2, keepdims=True)), axis=-1)
    

    
    if len(np.shape(im_in_3)) == 3:
        im_in_3 = np.mean(im_in_3[..., 0:3], axis = 2, keepdims=True)
    else:
        im_in_3 = np.stack([im_in_3], axis=2)
        
    im_in1 = np.concatenate((im_in_1, im_in_2, im_in_3), axis=2)
    
    im_label1[..., 1] = detectRed(im_out)
    im_label1[..., 0] = 1 - im_label1[..., 1]
    #
    # # convert to Data class
    data = data_net.Data(im_in1, im_label1, colors_sep=False, big_patches=width, ext=ext, bool_tophat=False)
    #
    return data

def test_data(set, width, ext):
    if set == 'zach':
        ex_raw_i = ex_raw_zach
    elif set == 'zach_small':
        ex_raw_i = ex_raw_zach_small
    elif set == 'hand':
        ex_raw_i = ex_raw_hand
    elif set == 'hand_small':
        ex_raw_i = ex_raw_hand_small
    elif set == 'hand_big':
        ex_raw_i = ex_raw_hand_big
    foo = ex_raw_i()
    return raw_data(foo, width, ext)