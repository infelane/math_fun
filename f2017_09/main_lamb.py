# main file from 2017_09_18

import numpy as np
import matplotlib.pyplot as plt

from f2017_06 import block_data2
from f2017_08.hsi import tools_data, tools_plot, tools_analysis
from f2017_09.lamb import nn
from link_to_soliton.paint_tools import image_tools
import link_to_keras_ipi as keras_ipi


class MainData(object):
    version = 6
    zoom = 1
    w = 10
    def __init__(self, version = None, set = 'hand_small', w = None, lr = None):
        self.set = set
        if set == 'hand_big':
            self.foo = block_data2.ex_raw_hand_big()
        elif set == 'hand_small':
            self.foo = block_data2.ex_raw_hand_small()
        elif set == 'zach_small':
            self.foo = block_data2.ex_raw_zach_small()
        elif set == 'zach_big':
            self.foo = block_data2.ex_raw_zach()
        else:
            raise ValueError('Unknown set')
        
        if version is not None:
            self.version = version
        if w is not None:
            self.w = w

        self.lr = lr
            
    def get_img_clean(self):
        return self.foo.im_in_1
    
    def get_img_rgb(self):
        return self.foo.im_in_2
    
    def get_img_ir(self):
        shape = np.shape(self.foo.im_in_3)
        
        if len(shape) == 3:
            if shape[2] == 3:
                return np.mean(self.foo.im_in_3, axis = 2, keepdims=True)
            elif shape[2] == 1:
                return self.foo.im_in_3
            else:
                raise ValueError
        
        elif len(shape) == 2:
            return np.reshape(self.foo.im_in_3, newshape=(shape[0], shape[1], 1))
        
        else:
            raise ValueError
    
    def get_img_y(self):
        folder_save = '/scratch/lameeus/data/ghent_altar/load/'
        
        if self.set == 'hand_big':
            file_name = 'y_annot_evangelist.npy'
        elif self.set == 'hand_small':
            file_name = 'y_annot_hand.npy'
        elif self.set == 'zach_small':
            file_name = 'y_annot_zach.npy'

        if 1:  # TODO SET AT 0 after done
            img = self.foo.im_out

            r0 = np.equal(img[:, :, 0], 0)
            r1 = np.equal(img[:, :, 0], 1)
            g0 = np.equal(img[:, :, 1], 0)
            g1 = np.equal(img[:, :, 1], 1)
            b0 = np.equal(img[:, :, 2], 0)
            b1 = np.equal(img[:, :, 2], 1)

            red = r1 * g0 * b0 #np.logical_and(np.logical_and(r1, g0), b0)
            blue = r0 * g0 * b1 #np.logical_and(np.logical_and(r0, g0), b1)

            shape = np.shape(img)
            shape_annot = [shape[0], shape[1], 2]

            img_annot = np.zeros(shape=shape_annot)

            img_annot[blue, 0] = 1
            img_annot[red, 1] = 1
      
            np.save(folder_save + file_name, img_annot)
        
        else:
            img_annot = np.load(folder_save + file_name)
        
        return img_annot


def main_training(dict_data, epochs = 10):
    network = nn.Network(version = dict_data.version, lr = dict_data.lr)
    network.load(make_backup=True)
    
    img_clean = dict_data.get_img_clean()
    img_ir = dict_data.get_img_ir()
    img_rgb = dict_data.get_img_rgb()
    img_y = dict_data.get_img_y()
    
    mask_annot_test = (np.greater(np.sum(img_y, axis = 2), 0)).astype(int)
    
    if 0:
        plt.imshow(img_y[:,:,0])
        plt.show()
    
    data = tools_data.Data(img_clean)
    
    x_list_test = data.img_mask_to_x([img_clean, img_ir, img_rgb, img_y], mask_annot_test,
                                     ext = [2, 2, 2, 0], name = 'lamb_test', bool_new= False)
    
    x_clean = x_list_test[0]
    x_rgb = x_list_test[2]
    x_ir = x_list_test[1]
    y = x_list_test[3]
    
    validation_split = 0.2
    network.train([x_clean, x_rgb, x_ir], y, epochs = epochs,
                  validation_split = validation_split, verbose=2)
    
    if 0:
        n_train = int((1 - validation_split) * (np.shape(x_clean))[0])
        x_clean_test = x_clean[n_train:, ...]
        x_rgb_test = x_rgb[n_train:, ...]
        x_ir_test = x_ir[n_train:, ...]
        y_test = y[n_train:, ...]
        
        hsi_pred = network.predict([x_clean_test, x_rgb_test, x_ir_test])
        acc_hsi = tools_analysis.categorical_accuracy(y_test, hsi_pred)
        print('hsi accuracy = {}%'.format(acc_hsi * 100))


def main_plotting(dict_data):
    network = nn.Network(version = dict_data.version, zoom = dict_data.zoom)
    network.load()
    network.summary()
    
    plot_version = 0

    img_clean = dict_data.get_img_clean()
    img_ir = dict_data.get_img_ir()
    img_rgb = dict_data.get_img_rgb()

    data = tools_data.Data(img_clean)
    data_ir = tools_data.Data(img_ir)

    x_clean = data.img_to_x(img_clean, ext=2)
    x_rgb = data.img_to_x(img_rgb, ext=2)
    x_ir = data_ir.img_to_x(img_ir, ext=2)

    if plot_version == 0:
        y_pred = network.predict([x_clean, x_rgb, x_ir])
        
    pred_img = data.y_to_img(y_pred)
    
    if 0:
        plt.figure()
        plt.imshow(dict_data.foo.im_out)
        plt.figure()
        print(np.shape(dict_data.get_img_y()))
        plt.imshow(dict_data.get_img_y()[..., 0])
        plt.figure()
        plt.imshow(dict_data.get_img_y()[..., 1])
        plt.show()
    
    if 1:
        metric = keras_ipi.metrics.dice_with_0_labels
        dice_score = tools_analysis.metrics_after_predict(metric, [dict_data.get_img_y(),pred_img])
        print("dice = {}".format(dice_score))
        metric = keras_ipi.metrics.jaccard_with_0_labels
        jaccard_score = tools_analysis.metrics_after_predict(metric, [dict_data.get_img_y(),pred_img])
        print("Jaccard = {}".format(jaccard_score))
        
    rgb = tools_plot.n_to_rgb(pred_img, with_sat = True, with_lum= True)
    pred_rgb = np.copy(img_clean)
    pred_rgb[pred_img[:,:, 1] > 0.5, :] = [1, 0, 0]
    
    if 1:
        image_tools.save_im(pred_img[:, :, 1], '/scratch/lameeus/data/ghent_altar/classification/class_{}_v{}.tif'.format(dict_data.set, dict_data.version))
        image_tools.save_im(pred_rgb, '/scratch/lameeus/data/ghent_altar/output/{}_v{}.tif'.format(dict_data.set, dict_data.version))

    tools_plot.imshow([img_clean, rgb, pred_rgb], title=['clean', 'certainty', 'prediction map'])
    plt.show()
    

def main():
    if 0:
        # versions = [1, 2, 3, 4, 6]
        versions = [2, 'combine', 'later']
        dict_data = MainData(lr = 1e-4)
        for run_i in range(10000):
            for version_i in versions:
                print('version {}'.format(version_i))
                dict_data.version = version_i
                main_training(dict_data, epochs=100)
                
    else:
        version = 6
        version = 'before' #1
        # version = 'combine'
        # version = 'later'
        # dict_data = MainData(set = 'hand_small', version = version, lr = 1e-4)
        # dict_data = MainData(set='hand_big', version = version)
        dict_data = MainData(set='zach_small', version = version)
        if 0:
            main_training(dict_data, epochs = 1000)
    
    main_plotting(dict_data)
    
    
if __name__ == '__main__':
    main()
