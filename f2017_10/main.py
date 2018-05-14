""" trying to implement multi scale segmentation """

import numpy as np
import matplotlib.pyplot as plt
import cv2

from f2017_08.hsi import tools_data, tools_plot
from f2017_09 import main_lamb
from f2017_10 import nn

import keras.callbacks

import warnings
warnings.simplefilter("always")
# def fxn():
#     warnings.warn("deprecated", DeprecationWarning)
# with warnings.catch_warnings():
#     warnings.simplefilter("always")
#     # fxn()


def get_xy(dict_data):
    img_clean = dict_data.get_img_clean()
    img_ir = dict_data.get_img_ir()
    img_rgb = dict_data.get_img_rgb()
    img_y = dict_data.get_img_y()
    mask_annot_test = (np.greater(np.sum(img_y, axis=2), 0)).astype(int)
    data = tools_data.Data(img_clean)
    
    # ext = [2, 2, 2, 0]
    # ext = [0, 0, 0, 0]
    ext = [3, 3, 3, 0]
    
    x_list_test = data.img_mask_to_x([img_clean, img_ir, img_rgb, img_y], mask_annot_test,
                                     ext=ext, name='lamb_stack_{}'.format(dict_data.set), bool_new=False)
    
    x_clean = x_list_test[0]
    x_rgb = x_list_test[2]
    x_ir = x_list_test[1]
    y = x_list_test[3]
    
    return x_clean, x_rgb, x_ir, y


version = 6
def main_plotting(dict_data):
    
    # if 1:
    #     # check dead relu's
    #     # import keras_contrib
    #     # import tensorflow.contrib
    #     import keras_contrib.callbacks.dead_relu_detector as dead_relu_detector
    
    zoom_max = 2 #5
    ext = 3
    w = dict_data.w
    
    img_clean = dict_data.get_img_clean()
    img_ir = dict_data.get_img_ir()
    img_rgb = dict_data.get_img_rgb()
    
    data = tools_data.Data(img_clean, w = w)

    pred_imgs = []
    for zoom_i in range(1, zoom_max+1):
        
        print(zoom_i)
        
        network = nn.Network(w=w, zoom=zoom_i)  #(version=version, zoom=zoom_i, w=w)
        network.load()

        # TODO blur image!
        if zoom_i != 1:
            sigma = zoom_i*2-1
            sigma_tuple = (sigma, sigma)
            img_clean_blur = cv2.GaussianBlur(img_clean, sigma_tuple, 0)
            img_rgb_blur = cv2.GaussianBlur(img_rgb, sigma_tuple, 0)
            img_ir_blur = cv2.GaussianBlur(img_ir,sigma_tuple, 0) # removes the last dim!
            img_ir_blur = np.expand_dims(img_ir_blur, axis = 2)
            
        else:
            img_clean_blur = img_clean
            img_rgb_blur = img_rgb
            img_ir_blur = img_ir

        ext_zoom = ext * zoom_i
        
        x_clean = data.img_to_x(img_clean_blur, ext=ext_zoom)
        x_rgb = data.img_to_x(img_rgb_blur, ext=ext_zoom)
        x_ir = data.img_to_x(img_ir_blur, ext=ext_zoom)
    
        y_pred = network.predict([x_clean, x_rgb, x_ir])
    
        pred_img_i = data.y_to_img(y_pred)

        # rgb_i = tools_plot.n_to_rgb(pred_img, with_sat=True, with_lum=True)

        pred_imgs.append(pred_img_i)
        
    conf = [np.max(pred_img_i, axis = -1) for pred_img_i in pred_imgs]

    pred_imgs_array = np.stack(pred_imgs)
    
    conf_array = np.stack(conf, axis = 0)
    
    print(np.shape(conf_array))
    
    select_max = np.argmax(conf_array, axis = 0)
    
    print(np.shape(select_max))

    pred_img_max = np.zeros(shape = np.shape(pred_imgs[0]))
    
    for i in range(zoom_max):
        pred_img_max[select_max == i, :] = pred_imgs_array[i, select_max == i, :]
    
    # pred_img_max = pred_imgs_array[select_max, :,:,:]
    
    def get_pred_rgb(img_clean, pred_img):
        cyan = [0, 1, 1]
        pred_rgb = np.copy(img_clean)
        pred_rgb[pred_img[:, :, 1] > 0.5, :] = cyan
        return pred_rgb

    pred_rgb = get_pred_rgb(img_clean, pred_img_max)
    
    select_map = tools_plot.n_to_rgb(select_max, anno_col=True, bool_argmax=False )
    
    def weighted_pred(pred_imgs_array):
        return np.mean(pred_imgs_array, axis = 0)

    pred_rgb_weighted = get_pred_rgb(img_clean, weighted_pred(pred_imgs_array))

    pred_rgb_s = []
    titles = []
    for i in range(zoom_max):
        # pred_rgb_i = np.copy(img_clean)
        # pred_rgb_i[pred_imgs[i][:,:, 1] > 0.5, :] = cyan

        pred_rgb_i = get_pred_rgb(img_clean, pred_imgs[i])
        
        pred_rgb_s.append(pred_rgb_i)
        titles.append('zoom {}'.format(i+1))

    tools_plot.imshow(pred_rgb_s, title=titles)
    
    tools_plot.imshow([pred_rgb, pred_rgb_weighted, select_map, pred_img_max[:,:,1]], title=['prediction map', 'weighted prediction map', 'selection map', 'prediction'])
    plt.show()
    
    
def main_training(dict_data = None):
    # zoom_i = 2
    network = nn.Network()#lr = 1e-2)
    network.load()
    
    # TODO train multi res at same time?
    
    dict_data_list = [main_lamb.MainData(set='hand_small', w = 50),
                      main_lamb.MainData(set='zach_small', w = 50)]
    
    x_clean = None
    x_rgb = None
    x_ir = None
    y = None
    
    for dict_data_i in dict_data_list:
        img_clean = dict_data_i.get_img_clean()
        img_ir = dict_data_i.get_img_ir()
        img_rgb = dict_data_i.get_img_rgb()
        img_y = dict_data_i.get_img_y()
        mask_annot_test = (np.greater(np.sum(img_y, axis=2), 0)).astype(int)
        data = tools_data.Data(img_clean)
    
        # ext = [2, 2, 2, 0]
        # ext = [0, 0, 0, 0]
        ext = [3, 3, 3, 0]

        x_list_test = data.img_mask_to_x([img_clean, img_ir, img_rgb, img_y], mask_annot_test,
                                         ext=ext, name='lamb_stack_{}'.format(dict_data_i.set), bool_new=False)

        if x_clean is None:
            # x_clean = x_list_test[0]
            # x_rgb = x_list_test[2]
            # x_ir = x_list_test[1]
            # y = x_list_test[3]

            x_clean, x_rgb, x_ir, y = get_xy(dict_data_i)
            
        else:
            x_clean = np.concatenate([x_clean, x_list_test[0]], axis = 0)
            x_rgb =np.concatenate([x_rgb, x_list_test[2]], axis = 0)
            x_ir = np.concatenate([x_ir, x_list_test[1]], axis = 0)
            y = np.concatenate([y, x_list_test[3]], axis = 0)
    
    network.train([x_clean, x_rgb, x_ir], y, epochs=10, save=False)
    
    
def main():
    # dict_data = main_lamb.MainData(set='hand_small', w = 50)
    dict_data = main_lamb.MainData(set='zach_small', w = 50)
    if 1:
        main_training(dict_data)
    
    main_plotting(dict_data)


if __name__ == '__main__':
    main()
