"""
Trying to copy his work for faster processing. Should be shared with him when finished!
"""



import numpy as np
import matplotlib.pyplot as plt
import time

from link_to_soliton.paint_tools import image_tools
from f2017_09.roman import nn
from f2017_08.hsi import tools_data, tools_plot


class MainData(object):
    def __init__(self, nr = 1):
        self.folder = '/ipi/private/lameeus/code/Matlab/roman_gui/'
        self.folder_data = '/home/lameeus/data/ghent_altar/roman/input/trainprog/'
        self.nr = nr

    def get_img_clean(self):
        if self.nr == 1:
            path = self.folder_data + 'rbp.png'
        
        elif self.nr == 2:
            path = self.folder + 'img/1.png'
        
        return image_tools.path2im(path)
    
    def get_img_grey(self):
        return np.mean(self.get_img_clean(), axis=2, keepdims=True)

    def get_img_xray(self):
        if self.nr == 1:
            path = self.folder_data + 'xbp.png'
        if self.nr == 2:
            path = self.folder + 'img/2.png'
        rgb = image_tools.path2im(path)
        return np.mean(rgb, axis = 2, keepdims=True)
    
    def get_img_ir(self):
        if self.nr == 1:
            path = self.folder_data + 'ibp.png'
        elif self.nr == 2:
            # TODO also ir for this
            print('incorrect modality!!!!')
            path = self.folder +'img/2.png'
            
        return image_tools.path2im(path)
    
    def get_annot(self):
        if self.nr == 2:
            path = self.folder + 'MaskaBackground.png'
            img_back = image_tools.path2im(path)
            path = self.folder + 'MaskaCrack.png'
            img_crack = image_tools.path2im(path)
    
            shape = np.shape(img_back)
        
            img_annot = np.zeros(shape = (shape[0], shape[1], 2), dtype=int)
            img_annot[img_back ==1, 0] = 1
            img_annot[img_crack == 1, 1] = 1
        
        elif self.nr == 1:
            path = self.folder_data + 'mbp.png'
            img_annot_rgb = image_tools.path2im(path)

            img_annot_grey = np.mean(img_annot_rgb, axis = 2)
            
            shape = np.shape(img_annot_grey)

            img_annot = np.zeros(shape = (shape[0], shape[1], 2), dtype=int)
            img_annot[img_annot_grey == 0, 0] = 1
            img_annot[img_annot_grey == 1, 1] = 1
            
        
        return img_annot
    
    def get_mask(self):
        path = self.folder + 'CNNmask.png'
        return image_tools.path2im(path)
    
def main_training(dict_data):
    network = nn.Network()
    if 1:
        network.load()

    img_grey = dict_data.get_img_grey()
    img_xray = dict_data.get_img_xray()
    img_ir = dict_data.get_img_ir()
    img_y = dict_data.get_annot()

    data = tools_data.Data(img_grey, w = 1)
    
    if 0:
        shape = np.shape(img_grey)
        mask = np.ones(shape = (shape[0], shape[1]))
        x_list_test = data.img_mask_to_x([img_grey, img_xray, img_ir, img_y],mask,
                                         ext=[7, 7, 7, 0], name='roman_paint_crack', bool_new=False)
    
    else:
        x_list_test = [data.img_to_x(img_i, ext_i) for img_i, ext_i in zip(
            [img_grey, img_xray, img_ir, img_y], [7, 7, 7, 0]) ]

    x = x_list_test[0:2]  # no ir yet
    y = x_list_test[3]

    network.train(x = x, y = y, epochs = 10)


def main_plotting(dict_data):
    t0 = time.time()
    
    w = 40
    ext = (7, 6)
    
    network = nn.Network(w = w, ext = ext)
    network.load()
    
    if 0:
        network.model.summary()
    
    img_grey = dict_data.get_img_grey()
    img_xray = dict_data.get_img_xray()
    img_ir = dict_data.get_img_ir()

    data = tools_data.Data(img_grey, w = w)

    x_clean = data.img_to_x(img_grey, ext=ext)
    x_xray = data.img_to_x(img_xray, ext=ext)
    x_ir = data.img_to_x(img_ir, ext=ext)

    t0_proc = time.time()
    
    y_pred = network.predict([x_clean, x_xray])
    pred_img = data.y_to_img(y_pred)
    
    t1_proc = time.time()
    total = t1_proc - t0_proc
    print("processing time: {} s".format(total))
    
    rgb = tools_plot.n_to_rgb(pred_img, with_sat=True, with_lum=True)

    img_clean = dict_data.get_img_clean()
    pred_rgb = np.copy(img_clean)
    pred_rgb[pred_img[:,:, 1] > 0.5, :] = [1, 0, 0]
    
    tools_plot.imshow([rgb, pred_rgb], title=['certainty', 'prediction map'])
    
    t1 = time.time()
    total = t1 - t0
    print("{} s".format(total))
    plt.show()
    

def main():
    dict_data = MainData()
    
    if 0:
        annot = dict_data.get_annot()
        plt.subplot(1, 2, 1)
        plt.imshow(annot[..., 0])
        plt.subplot(1, 2, 2)
        plt.imshow(annot[..., 1])
        plt.show()
    
    if 0:
        main_training(dict_data)
    
    if 1:
        dict_data = MainData(nr = 2)
        main_plotting(dict_data)

if __name__ == '__main__':
    main()
