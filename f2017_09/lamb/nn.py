import keras
from keras.layers import Conv2D, Input, concatenate
from keras.models import Model
import numpy as np

from f2017_08.hsi import tools_analysis

import link_to_keras_ipi as keras_ipi
from f2017_08.hsi.tools_network import gen_in

# depth = 4
depth_i = 50
kr = keras.regularizers.l2(1e-5)


def elu(filters, kernel_size, zoom = 1):
    return Conv2D(filters, kernel_size, activation='elu', kernel_regularizer=kr, dilation_rate=zoom)


def softmax(filters, kernel_size, zoom = 1):
    keras.activations.softmax
    return Conv2D(filters, kernel_size, activation='softmax', kernel_regularizer= kr, dilation_rate = zoom)


def gen_net1(layer_clean, layer_rgb, layer_ir, zoom = 1):
    layer_model = concatenate([layer_clean, layer_rgb, layer_ir])
    layer_model = elu(10, (5, 5), zoom)(layer_model)
    layer_model = softmax(2, (1, 1), zoom)(layer_model)
    return layer_model


def gen_net2(layer_clean, layer_rgb, layer_ir, zoom = 1):
    """ concatenate all """
    
    depth = depth_i * 4
    
    layer_model = concatenate([layer_clean, layer_rgb, layer_ir])
    layer_model = elu(depth, (3, 3), zoom)(layer_model)
    layer_model = elu(depth, (3, 3), zoom)(layer_model)
    layer_model = elu(depth_i, (1, 1), zoom)(layer_model)
    layer_model = softmax(2, (1, 1), zoom)(layer_model)
    return layer_model


def gen_net3(layer_clean, layer_rgb, layer_ir, zoom = 1):
    """ only clean """

    depth = depth_i*4
    
    layer_model = elu(depth, (3, 3), zoom)(layer_clean)
    layer_model = elu(depth, (3, 3), zoom)(layer_model)
    layer_model = elu(depth_i, (1, 1), zoom)(layer_model)
    layer_model = softmax(2, (1, 1), zoom)(layer_model)
    
    return layer_model


def gen_net4(layer_clean, layer_rgb, layer_ir, zoom = 1):
    """ concatenate only at end """
    depth = depth_i*4//3
    
    layer_model_clean = elu(depth, (3, 3), zoom)(layer_clean)
    layer_model_clean = elu(depth, (3, 3), zoom)(layer_model_clean)
    layer_mode_rgb = elu(depth, (3, 3), zoom)(layer_rgb)
    layer_mode_rgb = elu(depth, (3, 3), zoom)(layer_mode_rgb)
    layer_mode_ir = elu(depth, (3, 3), zoom)(layer_ir)
    layer_mode_ir = elu(depth, (3, 3), zoom)(layer_mode_ir)
    
    layer_model = concatenate([layer_model_clean, layer_mode_rgb, layer_mode_ir])
    layer_model = elu(depth_i, (1, 1), zoom)(layer_model)
    layer_model = softmax(2, (1, 1), zoom)(layer_model)
    
    return layer_model


def gen_net5(layer_clean, layer_rgb, layer_ir, zoom=1):
    """ concatenate all possibilities"""

    layer_model_clean = elu(depth_i, (3, 3), zoom)(layer_clean)
    layer_model_clean = elu(depth_i, (3, 3), zoom)(layer_model_clean)
    layer_mode_rgb = elu(depth_i, (3, 3), zoom)(layer_rgb)
    layer_mode_rgb = elu(depth_i, (3, 3), zoom)(layer_mode_rgb)
    layer_mode_ir = elu(depth_i, (3, 3), zoom)(layer_ir)
    layer_mode_ir = elu(depth_i, (3, 3), zoom)(layer_mode_ir)
    layer_mode_conc = concatenate([layer_clean, layer_rgb, layer_ir])
    layer_mode_conc = elu(depth_i, (3, 3), zoom)(layer_mode_conc)
    layer_mode_conc = elu(depth_i, (3, 3), zoom)(layer_mode_conc)
    
    layer_model = concatenate([layer_model_clean, layer_mode_rgb, layer_mode_ir, layer_mode_conc])
    layer_model = elu(depth_i, (1, 1), zoom)(layer_model)
    layer_model = softmax(2, (1, 1), zoom)(layer_model)
    
    return layer_model


def gen_net7(layer_clean, layer_rgb, layer_ir, zoom=1):
    depth_i = 50
    layer_mode_rgb = elu(2*depth_i, (3, 3), zoom)(layer_rgb)
    layer_mode_rgb = elu(2*depth_i, (3, 3), zoom)(layer_mode_rgb)
    layer_mode_ir = elu(2*depth_i, (3, 3), zoom)(layer_ir)
    layer_mode_ir = elu(2*depth_i, (3, 3), zoom)(layer_mode_ir)
    
    layer_model = concatenate([layer_mode_rgb, layer_mode_ir])
    layer_model = softmax(2, (1, 1), zoom)(layer_model)
    
    return layer_model

def gen_net8(layer_clean, layer_rgb, layer_ir, zoom=1):
    """ concatenate all possibilities"""
    
    depth = depth_i*4   # same depth as v2, but now for every channel (x4 amount of parameters)
    
    layer_model_clean = elu(depth, (3, 3), zoom)(layer_clean)
    layer_model_clean = elu(depth, (3, 3), zoom)(layer_model_clean)
    layer_mode_rgb = elu(depth, (3, 3), zoom)(layer_rgb)
    layer_mode_rgb = elu(depth, (3, 3), zoom)(layer_mode_rgb)
    layer_mode_ir = elu(depth, (3, 3), zoom)(layer_ir)
    layer_mode_ir = elu(depth, (3, 3), zoom)(layer_mode_ir)
    layer_mode_conc = concatenate([layer_clean, layer_rgb, layer_ir])
    layer_mode_conc = elu(depth, (3, 3), zoom)(layer_mode_conc)
    layer_mode_conc = elu(depth, (3, 3), zoom)(layer_mode_conc)
    
    layer_model = concatenate([layer_model_clean, layer_mode_rgb, layer_mode_ir, layer_mode_conc])
    layer_model = elu(depth_i, (1, 1), zoom)(layer_model)
    layer_model = softmax(2, (1, 1), zoom)(layer_model)
    
    return layer_model


def gen_net9(layer_clean, layer_rgb, layer_ir, zoom=1):
    """ concatenate only at end """
    depth = depth_i * 4
    
    layer_model_clean = elu(depth, (3, 3), zoom)(layer_clean)
    layer_model_clean = elu(depth, (3, 3), zoom)(layer_model_clean)
    layer_mode_rgb = elu(depth, (3, 3), zoom)(layer_rgb)
    layer_mode_rgb = elu(depth, (3, 3), zoom)(layer_mode_rgb)
    layer_mode_ir = elu(depth, (3, 3), zoom)(layer_ir)
    layer_mode_ir = elu(depth, (3, 3), zoom)(layer_mode_ir)
    
    layer_model = concatenate([layer_model_clean, layer_mode_rgb, layer_mode_ir])
    layer_model = elu(depth_i, (1, 1), zoom)(layer_model)
    layer_model = softmax(2, (1, 1), zoom)(layer_model)
    
    return layer_model


class Network(object):
    folder_model = '/scratch/lameeus/data/ghent_altar/net_weight/net_2017_09/'
    
    def __init__(self, version = 1, zoom = 1, lr = None, w = 10):
        """ zoom is for multi-resolution evalution (basically all stride is *zoom"""
        if lr is None:
            lr = 1e-4
        self.version = version
        ext = 2
        ext_zoom = zoom *ext
        layer_clean = gen_in(w, ext_zoom, 3, name = 'in_clean')
        layer_rgb = gen_in(w, ext_zoom, 3, name='in_rgb')
        layer_ir = gen_in(w, ext_zoom, 1, name='in_ir')

        if version == 1:
            self.file_name = 'w_v1_conc_cnn'
            layer_model = gen_net1(layer_clean, layer_rgb, layer_ir, zoom)
            
        elif version == 2:
            self.file_name = 'w_v2_conc_cnn'
            layer_model = gen_net2(layer_clean, layer_rgb, layer_ir, zoom)
            
        elif version == 3:
            self.file_name = 'w_v3_rgb'
            layer_model = gen_net3(layer_clean, layer_rgb, layer_ir, zoom)
            
        elif version == 4:
            self.file_name = 'w_v4_cnn_conc'
            layer_model = gen_net4(layer_clean, layer_rgb, layer_ir, zoom)
            
        elif version == 5:
            self.file_name = 'w_v5_cnn_conc'
            layer_model = gen_net5(layer_clean, layer_rgb, layer_ir, zoom)
            
        elif version == 6:
            self.file_name = 'w_v6_cnn_conc'
            layer_model = gen_net5(layer_clean, layer_rgb, layer_ir, zoom)
            
        elif version == 'before':
            self.file_name = 'w_v7_cnn_before'
            layer_model = gen_net7(layer_clean, layer_rgb, layer_ir, zoom)
            
        elif version == 'combine':
            self.file_name = 'w_v8_cnn_combine'
            layer_model = gen_net8(layer_clean, layer_rgb, layer_ir, zoom)
            
        elif version == 'later':
            self.file_name = 'w_v9_cnn_later'
            layer_model = gen_net9(layer_clean, layer_rgb, layer_ir, zoom)
            
        else:
            raise ValueError('not implemented version')

        loss = keras.losses.categorical_crossentropy
        # optimizer = {'class_name': 'adam', 'config': {'lr': lr}}
        optimizer = keras.optimizers.adam(lr = lr, clipnorm = 1.)
        
        dice = keras_ipi.metrics.dice_with_0_labels
        self.tb = keras.callbacks.TensorBoard(log_dir='/scratch/lameeus/data/ghent_altar/tensorboard/conc_net/', histogram_freq=0,
                                    write_graph=False, write_images=False)
               
        metrics = [dice]

        self.model = Model([layer_clean, layer_rgb, layer_ir], layer_model)
        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics = metrics
                           )

    def load(self, name = None, make_backup = False):
        if name is None:
            name = self.file_name
            
        self.model.load_weights(self.folder_model + name + '.h5')
        
        if make_backup:
            name_old = name + '_previous'
            self.model.save_weights(self.folder_model + name_old + '.h5')

    def save(self, name = None):
        if name is None:
            name = self.file_name
            
        # if name is None:
        #     self.model.save_weights(self.folder_model + self.file_name + '.h5')
        # else:
    
        self.model.save_weights(self.folder_model + name + '.h5')

    def train(self, x, y, epochs=1, save=True, validation_split = None, verbose = 1):
    
        callback = [self.tb]
        
        if save:
            cb_saver = keras.callbacks.LambdaCallback(on_epoch_end=lambda *a:self.save())
            callback.append(cb_saver)
            
        self.model.fit(x, y, epochs=epochs, validation_split=validation_split,
                       callbacks=callback, verbose=verbose)
                       
    def predict(self, x):
        return self.model.predict(x)
    
    def summary(self):
        return self.model.summary()