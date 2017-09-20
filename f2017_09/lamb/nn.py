import keras
from keras.layers import Conv2D, Input, concatenate
from keras.models import Model
import numpy as np

from f2017_08.hsi import tools_analysis
import link_to_keras_ipi as keras_ipi

def gen_in(w, ext, depth, name):
    shape = (w + 2 * ext, w + 2 * ext, depth)
    return Input(shape=shape, name=name)


class Network(object):
    folder_model = '/home/lameeus/data/ghent_altar/net_weight/net_17_09/'
    
    def __init__(self):
        w = 10
        ext = 2
        layer_clean = gen_in(w, ext, 3, name = 'in_clean')
        layer_rgb = gen_in(w, ext, 3, name='in_rgb')
        layer_ir = gen_in(w, ext, 1, name='in_ir')

        layer_model = concatenate([layer_clean, layer_rgb, layer_ir])
        layer_model = Conv2D(10, (5, 5), activation='elu')(layer_model)
        layer_model = Conv2D(2, (1, 1), activation='softmax')(layer_model)

        loss = keras.losses.categorical_crossentropy
        optimizer = {'class_name': 'adam', 'config': {'lr': 1.0e-4}}
        dice = keras_ipi.metrics.dice_with_0_labels
        metrics = [dice]

        self.model = Model([layer_clean, layer_rgb, layer_ir], layer_model)
        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics = metrics
                           )

    def load(self):
        name = 'w.h5'
        self.model.load_weights(self.folder_model + name)

    def save(self):
        name = 'w.h5'
        self.model.save_weights(self.folder_model + name)

    def train(self, x, y, epochs=1, save=True, validation_split = None):
        
        for i in range(epochs):
            print('Epoch {}/{}'.format(i + 1, epochs))
            self.model.fit(x, y, epochs=1, validation_split=validation_split)
            
            if save == True:
                self.save()
                
    def predict(self, x):
        return self.model.predict(x)
