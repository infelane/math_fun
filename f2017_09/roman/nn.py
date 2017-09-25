import keras
from keras.layers import Conv2D, Input, concatenate, MaxPooling2D
from keras.models import Model

import link_to_keras_ipi as keras_ipi
from f2017_08.hsi import tools_network
from f2017_08.hsi.tools_network import gen_in


kr = keras.regularizers.l2(0e-5)


def gen_net(layer_clean, layer_xray, zoom = 1):
    layer_model = concatenate([layer_clean, layer_xray])

    layer_model = Conv2D(6, (5, 5), activation='sigmoid', kernel_regularizer=kr,
                         name='first_conv2d', strides=zoom)(layer_model)
    layer_model = MaxPooling2D(strides=(1, 1))(layer_model)
    layer_model = Conv2D(12, (5, 5), activation='sigmoid', kernel_regularizer=kr,
                         name='second_conv2d', strides=zoom, dilation_rate = (2, 2)
                         )(layer_model)
    layer_model = Conv2D(2, (1, 1), activation='sigmoid', kernel_regularizer=kr,
                         name='final_conv2d', strides=zoom)(layer_model)
    
    return layer_model


class Network(object):
    folder_model = '/home/lameeus/data/ghent_altar/net_weight/net_roman/'
    file_name = 'roman.h5'
    
    def __init__(self, w = 1, ext = 7):
        dice = keras_ipi.metrics.dice_with_0_labels

        metrics = [dice]
        
        ext = ext

        layer_clean = gen_in(w, ext, 1, name = 'in_clean')
        layer_xray = gen_in(w, ext, 1, name='in_xray')
        
        layer_model = gen_net(layer_clean, layer_xray)
        
        loss = keras.losses.mean_squared_error
        optimizer = {'class_name': 'adam', 'config': {'lr': 1.0e-4}}
        
        self.model = Model([layer_clean, layer_xray], layer_model)
        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=metrics
                           )
        
    def load(self):
        self.model.load_weights(self.folder_model + self.file_name)

    def save(self):
        self.model.save_weights(self.folder_model + self.file_name)
        
    def predict(self, x):
        
 
        
        return self.model.predict(x)

   

    def train(self, x, y, epochs=1, save=True, validation_split=None):
        callbacks = []
        if save:
            cb_saver = keras.callbacks.LambdaCallback(on_epoch_end=lambda *a: self.save())
            callbacks.append(cb_saver)
    
        self.model.fit(x, y, epochs=epochs, validation_split=validation_split,
                       callbacks=callbacks)
