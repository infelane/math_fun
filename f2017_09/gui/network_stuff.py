""" Training and Inference are here """
import time
import numpy as np

from f2017_09 import main_lamb
from f2017_09.lamb import nn
from f2017_08.hsi import tools_data, tools_plot


class NetworkStuff(object):
    def __init__(self):
        self.dict_data = main_lamb.MainData(set = 'hand_small')
    
        self.network = nn.Network(version=self.dict_data.version, zoom=self.dict_data.zoom)
        
    def _init_network(self):
        self.network = nn.Network(version=self.dict_data.version, zoom=self.dict_data.zoom, lr = 1e-3)
        
    def load_network(self, name):
        if name == 'pretrained':
            self.network.load()
        elif name == 'none':
            self._init_network()
        elif name == 'demo 200':
            self.network.load(name='w_demo_ugent')
        elif name == 'bad':
            self.network.load(name='w_none')
        else:
            raise ValueError('unknown option name')
        
    def save_network_custom(self):
        self.network.save('w_demo_ugent')
        
    def loading_options(self):
        return ['none', 'pretrained', 'demo 200', 'bad']

    def set_name_set(self, name):
        if name == 'hand_big':
            self.dict_data_set = main_lamb.MainData(set='hand_big')
        elif name == 'hand_small':
            self.dict_data_set = main_lamb.MainData(set='hand_small')
    
    def training(self, epoch_func = None, func_print = print):
        print('started training')
        t0 = time.time()
        
        img_clean = self.dict_data.get_img_clean()
        img_ir = self.dict_data.get_img_ir()
        img_rgb = self.dict_data.get_img_rgb()
        img_y = self.dict_data.get_img_y()
        
        t1_proc = time.time()
        total = t1_proc - t0
        print("time loading: {} s".format(total))

        mask_annot_test = (np.greater(np.sum(img_y, axis=2), 0)).astype(int)

        data = tools_data.Data(img_clean)
    
        x_list_test = data.img_mask_to_x([img_clean, img_ir, img_rgb, img_y], mask_annot_test,
                                         ext=[2, 2, 2, 0], name='lamb_test', bool_new=False)
    
        t1_proc = time.time()
        total = t1_proc - t0
        print("time x: {} s".format(total))
    
        x_clean = x_list_test[0]
        x_rgb = x_list_test[2]
        x_ir = x_list_test[1]
        y = x_list_test[3]
    
        epochs = 5
        validation_split = 0.2
        
        for i in range(epochs):
            func_print('epoch {}/{}'.format(i+1, epochs))
            
            self.network.train([x_clean, x_rgb, x_ir], y, epochs=1, validation_split=validation_split, save = False)
            
            if epoch_func is not None:
                epoch_func()
                

        self.save_network_custom()

        t1_proc = time.time()
        total = t1_proc - t0
        print("time training: {} s".format(total))

        func_print('Training complete ')

    def inference(self):
        t0 = time.time()
    
        version = 0
    
        if 0:
            img_clean = self.dict_data.get_img_clean()
            img_ir = self.dict_data.get_img_ir()
            img_rgb = self.dict_data.get_img_rgb()
        else:
            img_clean = self.dict_data_set.get_img_clean()
            img_ir = self.dict_data_set.get_img_ir()
            img_rgb = self.dict_data_set.get_img_rgb()

        t1_proc = time.time()
        total = t1_proc - t0
        print("time loading: {} s".format(total))
    
        data = tools_data.Data(img_clean)
        data_ir = tools_data.Data(img_ir)
        
        t1_proc = time.time()
        total = t1_proc - t0
        print("time data: {} s".format(total))
    
        x_clean = data.img_to_x(img_clean, ext=2)
        x_rgb = data.img_to_x(img_rgb, ext=2)
        x_ir = data_ir.img_to_x(img_ir, ext=2)
        
        t1_proc = time.time()
        total = t1_proc - t0
        print("time x: {} s".format(total))
    
        if version == 0:
            y_pred = self.network.predict([x_clean, x_rgb, x_ir])
    
        pred_img = data.y_to_img(y_pred)
        
        t1_proc = time.time()
        total = t1_proc - t0
        print("time prediction: {} s".format(total))
    
        # if 1:
        #     metric = keras_ipi.metrics.dice_with_0_labels
        #     a = tools_analysis.metrics_after_predict(metric, [dict_data.get_img_y(), pred_img])
        #     print(a)
    
        # rgb = tools_plot.n_to_rgb(pred_img, with_sat=True, with_lum=True)
        pred_rgb = np.copy(img_clean)
        pred_rgb[pred_img[:, :, 1] > 0.5, :] = [1, 0, 0]
        #
        # if 1:
        #     image_tools.save_im(pred_img[:, :, 1], '/home/lameeus/data/ghent_altar/classification/class_hand_big.tif')
        #     image_tools.save_im(pred_rgb, '/home/lameeus/data/ghent_altar/output/hand_big.tif')
        #
        # tools_plot.imshow([rgb, pred_rgb], title=['certainty', 'prediction map'])
        # plt.show()

        t1_proc = time.time()
        total = t1_proc - t0
        print("time final: {} s".format(total))
        
        return pred_rgb
