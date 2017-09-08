""" the framework
"""

import numpy as np
import time
import matplotlib.pyplot as plt

from f2017_08.hsi import tools_datasets, tools_data, nn, tools_plot, tools_analysis
from f2017_08 import main_multimodal
from link_to_soliton.paint_tools import image_tools


def main_data():
    """ to split up script, I'll create a dictionary where I can store all data
    IDEA 2: make a class that only generates the data when needed!
    """
    
    dict_data = {}
    
    if 1:
        # annotations
        y_annot_img = tools_datasets.hsi_annot()
        
        if 0:
            y_annot_img = y_annot_img[500:-500:,500:-500, :]
        
        dict_data.update({'y_annot_img': y_annot_img})
        
        if 0:
            rgb_annot = tools_plot.n_to_rgb(y_annot_img)
            plt.imshow(rgb_annot)
            plt.show()

        del(y_annot_img)
        
    if 1:
        # Test annotation

        y_annot_img_test = tools_datasets.hsi_annot_test()
        dict_data.update({'y_annot_img_test': y_annot_img_test})

    if 1:
        # the hsi image
        img_norm = tools_datasets.hsi_processed()
        dict_data.update({'hsi_img': img_norm})
        
        del(img_norm)
        
    if 1:
        mask = tools_datasets.hsi_mask()
        print(np.shape(mask))
        dict_data.update({'mask_painting': mask})
        del (mask)
        
    if 1:
        shape = np.shape(dict_data['y_annot_img_test'])
        mask_annot_test = np.zeros((shape[0], shape[1]), dtype=int)
        bool = (np.sum(dict_data['y_annot_img_test'], axis=2) >= 1.0)
        mask_annot_test[bool] = 1
        dict_data.update({'mask_annot_test': mask_annot_test})
        del (mask_annot_test)
        
    if 1:
        shape = np.shape(dict_data['y_annot_img'])
        mask = np.zeros((shape[0], shape[1]), dtype=int)
        bool = (np.sum(dict_data['y_annot_img'], axis = 2) >= 1.0)
        mask[bool] = 1
        dict_data.update({'mask_annot': mask})
        del (mask)
        
    if 1:
        rgb = image_tools.path2im('/home/lameeus/data/hsi/rgb_registrated.png')
        dict_data.update({'rgb' : rgb})
        del (rgb)
        
    return dict_data


def main_training(dict_data):
    network = nn.Network()
    network.load()
    
    # Things to load
    data = tools_data.Data(dict_data['hsi_img'])
    mask_annot = dict_data['mask_annot']
    mask_painting = dict_data['mask_painting']
    mask_annot_test = dict_data['mask_annot_test']
    hsi_img = dict_data['hsi_img']
    y_annot_img = dict_data['y_annot_img']
    y_annot_img_test = dict_data['y_annot_img_test']
    rgb = dict_data['rgb']
    
    x_list = data.img_mask_to_x([hsi_img, y_annot_img], mask_annot, ext = [2, 0])
    x_train = x_list[0]
    y_annot = x_list[1]
    
    x_rgb = data.img_mask_to_x([rgb], mask_annot, ext = [2],
                               name = 'rgb', bool_new= False)

    x_list_test = data.img_mask_to_x([hsi_img, y_annot_img_test, rgb], mask_annot_test, ext = [2, 0, 2], name = 'rgb_test', bool_new= True)
    x_test = x_list_test[0]
    y_annot_test = x_list_test[1]
    x_rgb_test = x_list_test[2]
    
    x_auto = data.img_mask_to_x(hsi_img, mask_painting, ext = 2, name ='auto')

    epochs = 0
    if epochs:
        network.train_auto(x_auto, epochs=epochs, save=True)

    epochs = 0
    if epochs:
        network.train_discr(x_train, y_annot, epochs=epochs, save=True)
    
    epochs = 0
    if epochs:
        network.train_all(x_train, y_annot, epochs=epochs, save=True)
        
    epochs = 0
    if epochs:
        network.train_rgb(x_rgb, y_annot, epochs = epochs, save=True)
        
    epochs = 0
    if epochs:
        network.train_class(hsi = x_train, rgb = x_rgb, annot = y_annot, epochs = epochs, save = True)
    
    if 1:
        hsi_pred = network.predict_class_hsi(x_test)
        acc_hsi = tools_analysis.categorical_accuracy(y_annot_test, hsi_pred)
        print('hsi accuracy = {}%'.format(acc_hsi * 100))

        rgb_pred = network.predict_class_rgb(x_rgb_test)
        acc_rgb = tools_analysis.categorical_accuracy(y_annot_test, rgb_pred)
        print('rgb accuracy = {}%'.format(acc_rgb * 100))
        
    network.stop()

def main_processing(dict_data):
    dict_data_proc = {}
    
    if 1:
        dict_data_proc.update({'hsi_img':dict_data['hsi_img']})
        dict_data_proc.update({'mask_annot':dict_data['mask_annot']})
        dict_data_proc.update({'mask_painting': dict_data['mask_painting']})
        
    network = nn.Network()
    network.load()
    data = tools_data.Data(dict_data['hsi_img'])

    x_ext2 = data.img_to_x(dict_data['hsi_img'], ext=2)

    data_rgb = tools_data.Data(dict_data['rgb'])
    x_rgb = data_rgb.img_to_x(dict_data['rgb'], ext = 2)

    if 0:
        y = network.predict_auto(x_ext2)
        y_img = data.y_to_img(y)
        dict_data_proc.update({'auto_img': y_img})
        
    if 0:
        # Code
        code = network.predict_code(x_ext2)
        code_img = data.y_to_img(code, ext=1)
        dict_data_proc.update({'code_img': code_img})
        
    if 0:
        y_pred = network.predict_discr(x_ext2)
        pred_img = data.y_to_img(y_pred)
        dict_data_proc.update({'pred_img': pred_img})
        
    if 0:
        y_pred = network.predict_rgb(x_rgb)
        pred_rgb = data.y_to_img(y_pred)
        dict_data_proc.update({'pred_rgb': pred_rgb})
        
    if 1:
        y_pred_hsi = network.predict_class_hsi(x_ext2)
        y_pred_rgb = network.predict_class_rgb(x_rgb)
        pred_hsi = data.y_to_img(y_pred_hsi)
        pred_rgb = data.y_to_img(y_pred_rgb)
        dict_data_proc.update({'pred_class_hsi': pred_hsi})
        dict_data_proc.update({'pred_class_rgb': pred_rgb})
        
    network.stop()
    
    return dict_data_proc


def main_plotting(dict_data_proc):
    if 0:
        img_norm = dict_data_proc['hsi_img']
        rgb1 = hsi_2_rgb(img_norm)

        img_norm = dict_data_proc['auto_img']
        rgb2 = hsi_2_rgb(img_norm)
        tools_plot.imshow([rgb1, rgb2], mask=dict_data_proc['mask_painting'], n = 2, title=['hsi rgb', 'auto rgb'])

        folder = '/home/lameeus/data/hsi/'
        if 1:
            image_tools.save_im(rgb1, folder + 'hsi_rgb.png')
        if 1:
            image_tools.save_im(rgb2, folder + 'hsi_auto_rgb.png')
    
    if 0:
        code_img = dict_data_proc['code_img']
        rgb = tools_plot.n_to_rgb(code_img)
        rgb_p = tools_plot.n_to_rgb(code_img, with_sat = True, with_lum= True)
        tools_plot.imshow([rgb, rgb_p], n = 2, mask=dict_data_proc['mask_annot'], title=['arg(p_max)', 'p_max'])
        tools_plot.imshow([rgb, rgb_p], n = 2, mask=dict_data_proc['mask_painting'], title=['arg(p_max)', 'p_max'])
    
    if 0:
        pred_img = dict_data_proc['pred_img']
        rgb = tools_plot.n_to_rgb(pred_img)
        rgb_p = tools_plot.n_to_rgb(pred_img, with_sat = True, with_lum= True)
        tools_plot.imshow([rgb, rgb_p], n=2, mask=dict_data_proc['mask_annot'], title=['arg(segm)', 'semg_max'])
        tools_plot.imshow([rgb, rgb_p], n=2, mask=dict_data_proc['mask_painting'], title=['arg(segm)', 'semg_max'])

    if 0:
        pred_img = dict_data_proc['pred_rgb']
        rgb = tools_plot.n_to_rgb(pred_img)
        rgb_p = tools_plot.n_to_rgb(pred_img, with_sat = True, with_lum= True)
        tools_plot.imshow([rgb, rgb_p], n=2, mask=dict_data_proc['mask_painting'], title=['arg(segm rgb)', 'semg_max rgb'])

    if 1:
        pred_img = dict_data_proc['pred_class_hsi']
        rgb = tools_plot.n_to_rgb(pred_img, anno_col= True)
        rgb_p = tools_plot.n_to_rgb(pred_img, with_sat = True, with_lum= True)
        tools_plot.imshow([rgb, rgb_p], n=2, mask=dict_data_proc['mask_painting'], title=['arg(segm hsi)', 'semg_max hsi'])
        
        folder = '/home/lameeus/data/hsi/outputs/'
        image_tools.save_im(rgb, path= folder + 'class_hsi.tif', check_prev=True)
        
        pred_img = dict_data_proc['pred_class_rgb']
        rgb = tools_plot.n_to_rgb(pred_img, anno_col= True)
        rgb_p = tools_plot.n_to_rgb(pred_img, with_sat = True, with_lum= True)
        tools_plot.imshow([rgb, rgb_p], n=2, mask=dict_data_proc['mask_painting'], title=['arg(segm rgb)', 'semg_max rgb'])

        image_tools.save_im(rgb, path=folder + 'class_rgb.tif', check_prev=True)


    plt.show()


def hsi_2_rgb(a):
    hsi_data = main_multimodal.HsiData()
    return hsi_data.to_rgb(a)


def main():
    t0 = time.time()
    
    dict_data = main_data()
    
    if 1:
        main_training(dict_data)
        
    dict_data_proc = main_processing(dict_data)
    
    t1 = time.time()
    total = t1 - t0
    print(total)

    main_plotting(dict_data_proc)
    

if __name__ == '__main__':
    main()