"""
All loaders of data
"""

from keras.utils import to_categorical
import numpy as np
from numpy import random

from link_to_soliton.paint_tools import image_tools
from f2018_01.tools import t_datasets


def load_art(ext_tot=0):
    from f2017_08.hsi import tools_data, tools_plot, tools_analysis
    from f2017_09 import main_lamb
    
    sets = ['hand_small', 'zach_small']
    
    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []
    x_val_list = []
    y_val_list = []
    
    for set_i in sets:
        foo = main_lamb.MainData(set=set_i)
        
        img_clean = foo.get_img_clean()
        img_ir = foo.get_img_ir()
        img_rgb = foo.get_img_rgb()
        img_y = foo.get_img_y()
        
        mask_train, mask_test, mask_val = checkers_masks(img_y)
        
        # mask_annot_test = (np.greater(np.sum(img_y, axis=2), 0)).astype(int)
        
        def sub_from_mask(img_y_a, mask):
            img_y_sub_a = np.zeros(np.shape(img_y_a))
            img_y_sub_a[mask == 1, :] = img_y_a[mask == 1, :]
            return img_y_sub_a
        
        img_y_train = sub_from_mask(img_y, mask_train)
        img_y_test = sub_from_mask(img_y, mask_test)
        img_y_val = sub_from_mask(img_y, mask_val)
        
        mask_annot_train = (np.greater(np.sum(img_y_train, axis=2), 0)).astype(int)
        mask_annot_test = (np.greater(np.sum(img_y_test, axis=2), 0)).astype(int)
        mask_annot_val = (np.greater(np.sum(img_y_val, axis=2), 0)).astype(int)
        
        total_points = np.sum(mask_annot_train) + np.sum(mask_annot_test) + np.sum(mask_annot_val)
        print('total points: {}'.format(total_points))
    
        data = tools_data.Data(img_clean)
        
        ext=ext_tot//2
        
        for subset in ['train', 'test', 'val']:
            img_y_sub = None
            mask_annot_sub = None
            if subset == 'train':
                img_y_sub = img_y_train
                mask_annot_sub = mask_annot_train
            elif subset == 'test':
                img_y_sub = img_y_test
                mask_annot_sub = mask_annot_test
            elif subset == 'val':
                img_y_sub = img_y_val
                mask_annot_sub = mask_annot_val
                # mask_annot_sub = mask_test
                
            if 0:
                # basically for debugging when needed
                import matplotlib.pyplot as plt
                plt.figure()
                plt.imshow(mask_annot_sub)
                plt.show()
            
            name = 'lamb_unet_2018_{}_{}'.format(set_i, subset)
            x_list_sub = data.img_mask_to_x([img_clean, img_rgb, img_ir, img_y_sub], mask_annot_sub,
                                            ext=[ext, ext, ext, 0], name=name, bool_new=False)
            
            x_clean_sub = x_list_sub[0]
            x_rgb_sub = x_list_sub[1]
            x_ir_sub = x_list_sub[2]
            y_sub = x_list_sub[3]
            
            x_sub = [x_clean_sub, x_rgb_sub, x_ir_sub]
            
            if subset == 'train':
                x_train_list.append(x_sub)
                y_train_list.append(y_sub)
            elif subset == 'test':
                x_test_list.append(x_sub)
                y_test_list.append(y_sub)
            elif subset == 'val':
                x_val_list.append(x_sub)
                y_val_list.append(y_sub)
    
    def concat_list(x_list):
        x_list_trans = list(map(list, zip(*x_list)))
        
        return [np.concatenate(x_list_i, axis=0) for x_list_i in x_list_trans]
    
    x_train = concat_list(x_train_list)
    y_train = np.concatenate(y_train_list, axis=0)
    x_test = concat_list(x_test_list)
    y_test = np.concatenate(y_test_list, axis=0)
    x_val = concat_list(x_val_list)
    y_val = np.concatenate(y_val_list, axis=0)
    
    return (x_train, y_train), (x_test, y_test),  (x_val, y_val)


def load_mnist():
    """
    Loads the mnist dataset
    :return:
    """
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


def load_multi():
    folder_data = '/home/lameeus/data/2018_IEEE_challenge/2018_Release_Phase1/2018_Release_Phase1/'
    rgb_folder = 'VHR - RGB/'

    name1 = 'UH_NAD83_272056_3289689.tif'
    
    path1 = folder_data + rgb_folder + name1
    
    rgb = image_tools.path2im(path1)
    
    folder_ground = 'GT/'
    path_ground = folder_data + folder_ground
    name_header = '2018_IEEE_GRSS_DFC_GT_TR.hdr'
    name_image = '2018_IEEE_GRSS_DFC_GT_TR'

    img_y = t_datasets.hsi_raw(folder=path_ground, name_header=name_header,
                           name_image=name_image)

    img_y_correct = t_datasets.to_categorical_0_unknown(img_y[..., 0])

    img_y_correct_crop = img_y_correct[:,:1192, :]

    if 1:
        from f2017_08.hsi import tools_data
        data_x = tools_data.Data(rgb, w=10)
        data_y = tools_data.Data(img_y_correct_crop, w=1)

        x_all = data_x.img_to_x2(rgb, ext=0)
        y_all = data_y.img_to_x2(img_y_correct_crop, ext=0)

        n_all_x = np.shape(x_all)[0]
        n_all_y = np.shape(y_all)[0]
        
        assert n_all_x == n_all_y, '{} !+ {}'.format(n_all_x, n_all_y)
        
        n_train = int(0.8*n_all_x)
    
    x_train = x_all[:n_train, ...]
    y_train = y_all[:n_train, ...]

    x_test = x_all[n_train:, ...]
    y_test = y_all[n_train:, ...]
    
    return (x_train, y_train), (x_test, y_test)


def checkers_masks(img, split_ratio=0.8, checker_width=20, seed=314):
    """
    :param img:
    :param split_ratio: training/test ratio, validation is added
    :param checker_width: size of a checker square
    :return:
    """
    random.seed(seed)  # to make sure it's always the same output
    
    shape = np.shape(img)
    
    mask_train = np.zeros(shape=shape[0:2])
    mask_test = np.zeros(shape=shape[0:2])
    mask_val = np.zeros(shape=shape[0:2])
    
    split_ratio_val = (split_ratio + 1.0) / 2.
    
    h, w = shape[0: 2]
    
    def calc_n(a):
        return int(np.ceil(a / checker_width))
    
    n_h = calc_n(h)
    n_w = calc_n(w)
    
    bool_mask = random.uniform(size=(n_h, n_w))
    
    for i_h in range(n_h):
        for i_w in range(n_w):
            i_h_0 = i_h * checker_width
            i_h_1 = (i_h + 1) * checker_width
            i_w_0 = i_w * checker_width
            i_w_1 = (i_w + 1) * checker_width
            
            if bool_mask[i_h, i_w] < split_ratio:
                mask_train[i_h_0: i_h_1, i_w_0:i_w_1] = 1
            elif split_ratio <= bool_mask[i_h, i_w] < split_ratio_val:
                mask_test[i_h_0: i_h_1, i_w_0:i_w_1] = 1
            elif split_ratio_val <= bool_mask[i_h, i_w]:
                mask_val[i_h_0: i_h_1, i_w_0:i_w_1] = 1
            else:
                raise ValueError('This case is not well implemented')
                
    return mask_train, mask_test, mask_val
