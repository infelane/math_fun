""" the framework
"""

import numpy as np
import time
import matplotlib.pyplot as plt

from f2017_08.hsi import tools_datasets, tools_data, nn, tools_plot
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
        shape = np.shape(dict_data['y_annot_img'])
        mask = np.zeros((shape[0], shape[1]), dtype=int)
        bool = (np.sum(dict_data['y_annot_img'], axis = 2) >= 1.0)
        mask[bool] = 1
        dict_data.update({'mask_annot': mask})
        del (mask)
        
    return dict_data
    

def main_processing(dict_data):
    dict_data_proc = {}
    
    if 1:
        dict_data_proc.update({'hsi_img':dict_data['hsi_img']})
        dict_data_proc.update({'mask_annot':dict_data['mask_annot']})
        dict_data_proc.update({'mask_painting': dict_data['mask_painting']})
        
    network = nn.Network()
    network.load()
    data = tools_data.Data(dict_data['hsi_img'])
    
    if 1:
        x_ext2 = data.img_to_x(dict_data['hsi_img'], ext=2)
        y = network.predict_auto(x_ext2)
        y_img = data.y_to_img(y)
        dict_data_proc.update({'auto_img': y_img})
        
    if 1:
        # Code
        code = network.predict_code(x_ext2)
        code_img = data.y_to_img(code, ext=1)
        dict_data_proc.update({'code_img': code_img})
        
    if 1:
        y_pred = network.predict_discr(x_ext2)
        pred_img = data.y_to_img(y_pred)
        dict_data_proc.update({'pred_img': pred_img})
    
    network.stop()
    
    return dict_data_proc


def main_plotting(dict_data_proc):
    if 1:
        img_norm = dict_data_proc['hsi_img']
        rgb1 = hsi_2_rgb(img_norm)

        if 1:
            image_tools.save_im(rgb1, '/ipi/research/lameeus/data/hsi/hsi_rgb.png')

        img_norm = dict_data_proc['auto_img']
        rgb2 = hsi_2_rgb(img_norm)
        tools_plot.imshow([rgb1, rgb2], mask=dict_data_proc['mask_painting'], n = 2, title=['hsi rgb', 'auto rgb'])
        
    if 1:
        code_img = dict_data_proc['code_img']
        rgb = tools_plot.n_to_rgb(code_img)
        rgb_p = tools_plot.n_to_rgb(code_img, with_sat = True, with_lum= True)
        tools_plot.imshow([rgb, rgb_p], n = 2, mask=dict_data_proc['mask_annot'], title=['arg(p_max)', 'p_max'])
        tools_plot.imshow([rgb, rgb_p], n = 2, mask=dict_data_proc['mask_painting'], title=['arg(p_max)', 'p_max'])

    
    if 1:
        pred_img = dict_data_proc['pred_img']
        rgb = tools_plot.n_to_rgb(pred_img)
        rgb_p = tools_plot.n_to_rgb(pred_img, with_sat = True, with_lum= True)
        tools_plot.imshow([rgb, rgb_p], n=2, mask=dict_data_proc['mask_annot'], title=['arg(segm)', 'semg_max'])
        # rgb = tools_plot.n_to_rgb(pred_img)
        # rgb_p = tools_plot.n_to_rgb(pred_img, with_sat = True, with_lum= True)
        tools_plot.imshow([rgb, rgb_p], n=2, mask=dict_data_proc['mask_painting'], title=['arg(segm)', 'semg_max'])

    plt.show()


def hsi_2_rgb(a):
    hsi_data = main_multimodal.HsiData()
    return hsi_data.to_rgb(a)


def main():
    t0 = time.time()
    
    dict_data = main_data()
    y_annot_img = dict_data['y_annot_img']
    img_norm = dict_data['hsi_img']
    
    dict_data_proc = main_processing(dict_data)
    
    t1 = time.time()
    total = t1 - t0
    print(total)

    main_plotting(dict_data_proc)
   
    assert 1 == 0
    
    x_ext1 = data.img_to_x(img_norm, ext = 1)   # or x_ext2[:,1:-1, 1:-1, :]
    
    # import maus.paint_tools.image_tools
    # mask = maus.paint_tools.image_tools.path2im('/home/lameeus/data/hsi/mask.png')

    if 0:
        ...

    else:
        ...

    # x_train = data.img_mask_to_x(img_norm, mask, ext = 2)
    # y_annot = data.img_mask_to_x(y_annot_img, mask, ext=2)[:,2:-2,2:-2,:]
    
    x_list = data.img_mask_to_x([img_norm, y_annot_img], mask, ext = [2, 0])
    x_train = x_list[0]
    y_annot = x_list[1]

    print(np.shape(x_train))
    
    # x_small = x[0:10000, ...]
    
    print(np.shape(x_ext2))
    
    # print(np.max(img_norm))
    print(np.shape(img_norm))
    
    epochs = 0
    if epochs:
        network.train_auto(x_train, epochs = epochs, save = True)
    
    epochs = 0
    if epochs:
        network.train_discr(x_train, y_annot, epochs = epochs, save = True)

    epochs = 0
    if epochs:
        network.train_all(x_train, y_annot, epochs=epochs, save=True)
        
    bool_auto = False
    if bool_auto:
        y = network.predict_auto(x_ext2)
    else:
        y = network.predict_discr(x_ext2)
    
    

    network.stop()
    
    
    
    if bool_auto:
        y_rgb = hsi_data.to_rgb(y_img)
        tools_plot.imshow(y_rgb, title='auto encoded')
    
        if 0:
            image_tools.save_im(y_rgb, '/ipi/research/lameeus/data/hsi/y_rgb.png')


    
    # rgb = tools_plot.n_to_rgb(code_img, with_lum=True)
    # plt.figure()
    # plt.imshow(rgb)

    # rgb = tools_plot.n_to_rgb(code_img, with_sat=True)
    # plt.figure()
    # plt.imshow(rgb)

    rgb = tools_plot.n_to_rgb(code_img, with_sat=True, with_lum=True)
    tools_plot.imshow(rgb, title = 'arg(p_max) and p_max')
    
    rgb = tools_plot.n_to_rgb(code_img, with_col = False, with_sat=True, with_lum=True)
    tools_plot.imshow(rgb, mask = mask, title = 'p_max')
    
    code_rgb = code_img[:,:,0:3]
    min = np.min(code_rgb)
    max = np.max(code_rgb)
    code_rgb = (code_rgb-min)/(max - min)
    
    plt.figure()
    plt.imshow(code_rgb)
    
    plt.show()
    
    print(np.shape(y))
    print(np.shape(y_img))
    

if __name__ == '__main__':
    main()