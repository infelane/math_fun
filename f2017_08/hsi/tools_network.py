"""
constantly reused network structures
"""

import keras
from keras.layers import Conv2D, Input, Concatenate, Cropping2D

def gen_in(w:int, ext:int or tuple, depth:int, name):
    if type(ext) is tuple:
        shape = (w + ext[0] + ext[1], w + ext[0] + ext[1], depth)
    if type(ext) is int:
        shape = (w + 2 * ext, w + 2 * ext, depth)
    return Input(shape=shape, name=name)


def elu(filters, kernel_size, zoom = 1, kr = keras.regularizers.l2(1e-5)):
    return Conv2D(filters, kernel_size, activation='elu', kernel_regularizer=kr, dilation_rate=zoom)


def relu(filters, kernel_size, zoom = 1, kr = keras.regularizers.l2(1e-5)):
    return Conv2D(filters, kernel_size, activation='relu', kernel_regularizer=kr, dilation_rate=zoom)


def softmax(filters, kernel_size, zoom = 1, kr = keras.regularizers.l2(1e-5)):
    softmax = keras.activations.softmax
    return Conv2D(filters, kernel_size, activation=softmax, kernel_regularizer= kr, dilation_rate = zoom)


def inception(inputs, filters, kernel_sizes = ((1, 1), (3, 3), (5, 5)), zoom = 1, kr = keras.regularizers.l2(1e-5), activation ='elu'):
    
    kernel_w = [kernel_size_i[0] for kernel_size_i in kernel_sizes]
    kernel_w_max = max(kernel_w)
    
    convs = []
    for kernel_size_i in kernel_sizes:
        
        diff = [(kernel_w_max - kernel_size_ii)*zoom for kernel_size_ii in kernel_size_i]
        
        cropping00 = [a//2 for a in diff]
        cropping01 = [a - b for a,b in zip(diff, cropping00)]#) diff - cropping00
        
        cropping = ((cropping00[0], cropping01[0]), (cropping00[1], cropping01[1]))
        
        crop_i = Cropping2D(cropping)(inputs)
        
        convs.append(Conv2D(filters, kernel_size_i, activation=activation, kernel_regularizer=kr, dilation_rate=zoom)(crop_i))

    return Concatenate(axis = -1)(convs)


class BaseNetwork(object):
    callback = []
    def __init__(self, file_name, model, folder_model):
        # Overwrite!
        self.model = model
        self.file_name = file_name
        self.folder_model = folder_model
    
    def load(self, name=None, make_backup=False):
        if name is None:
            name = self.file_name
        
        self.model.load_weights(self.folder_model + name + '.h5')
        
        if make_backup:
            name_old = name + '_previous'
            self.model.save_weights(self.folder_model + name_old + '.h5')
    
    def save(self, name=None):
        if name is None:
            name = self.file_name
        
        self.model.save_weights(self.folder_model + name + '.h5')
    
    def train(self, x, y, epochs=1, save=True, validation_split=None, verbose=1):
        
        callback = self.callback
        
        if save:
            cb_saver = keras.callbacks.LambdaCallback(on_epoch_end=lambda *a: self.save())
            callback.append(cb_saver)
        
        self.model.fit(x, y, epochs=epochs, validation_split=validation_split,
                       callbacks=callback, verbose=verbose)
    
    def predict(self, x):
        return self.model.predict(x, verbose=1)
    
    def summary(self):
        return self.model.summary()
