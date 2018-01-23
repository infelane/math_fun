from keras.utils import to_categorical
import numpy as np
from numpy import random


def load_art(ext_tot=0):
    from f2017_06 import block_data2
    from f2017_08.hsi import tools_data, tools_plot, tools_analysis
    from f2017_09 import main_lamb
    # foo = block_data2.ex_raw_hand_small()
    
    sets = ['hand_small', 'zach_small']
    
    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []
    
    for set_i in sets:
        foo = main_lamb.MainData(set=set_i)
        
        img_clean = foo.get_img_clean()
        img_ir = foo.get_img_ir()
        img_rgb = foo.get_img_rgb()
        img_y = foo.get_img_y()
        
        mask_train, mask_test = checkers_masks(img_y)
        
        # mask_annot_test = (np.greater(np.sum(img_y, axis=2), 0)).astype(int)
        
        def sub_from_mask(img_y_a, mask):
            img_y_sub_a = np.zeros(np.shape(img_y_a))
            img_y_sub_a[mask == 1, :] = img_y_a[mask == 1, :]
            return img_y_sub_a
        
        img_y_train = sub_from_mask(img_y, mask_train)
        img_y_test = sub_from_mask(img_y, mask_test)
        
        mask_annot_train = (np.greater(np.sum(img_y_train, axis=2), 0)).astype(int)
        mask_annot_test = (np.greater(np.sum(img_y_test, axis=2), 0)).astype(int)
        
        data = tools_data.Data(img_clean)
        
        ext=ext_tot//2
        
        for subset in ['train', 'test']:
            img_y_sub = None
            mask_annot_sub = None
            if subset == 'train':
                img_y_sub = img_y_train
                # TODO remove correct
                mask_annot_sub = mask_annot_train
                # mask_annot_sub = mask_train
            elif subset == 'test':
                img_y_sub = img_y_test
                # TODO remove correct
                mask_annot_sub = mask_annot_test
                # mask_annot_sub = mask_test
                
            if 0:
                # basically for debugging when needed
                import matplotlib.pyplot as plt
                plt.figure()
                plt.imshow(mask_annot_sub)
                plt.show()
            
            name = 'lamb_unet_2018_{}_{}'.format(set_i, subset)
            x_list_sub = data.img_mask_to_x([img_clean, img_rgb, img_ir, img_y_sub], mask_annot_sub,
                                            ext=[ext, ext, ext, 0], name=name, bool_new=True)
            
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
    
    def concat_list(x_list):
        x_list_trans = list(map(list, zip(*x_list)))
        
        return [np.concatenate(x_list_i, axis=0) for x_list_i in x_list_trans]
    
    x_train = concat_list(x_train_list)
    y_train = np.concatenate(y_train_list, axis=0)
    x_test = concat_list(x_test_list)
    y_test = np.concatenate(y_test_list, axis=0)
    
    return (x_train, y_train), (x_test, y_test)


# Remove! or atleast alter dataset loader
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


def checkers_masks(img, split_ratio=0.8, checker_width=20, seed=314):
    """
    :param img:
    :param split_ratio: training/test ratio
    :param checker_width: size of a checker square
    :return:
    """
    random.seed(seed)  # to make sure it's always the same output
    
    shape = np.shape(img)
    
    mask_train = np.empty(shape=shape[0:2])
    
    h, w = shape[0: 2]
    
    def calc_n(a):
        return int(np.ceil(a / checker_width))
    
    n_h = calc_n(h)
    n_w = calc_n(w)
    
    bool_mask = random.uniform(size=(n_h, n_w))
    
    for i_h in range(n_h):
        for i_w in range(n_w):
            mask_train[i_h * checker_width:(i_h + 1) * checker_width, i_w * checker_width:(i_w + 1) * checker_width] = \
                1 if bool_mask[i_h, i_w] < split_ratio else 0
    
    mask_test = 1 - mask_train
    
    return mask_train, mask_test
