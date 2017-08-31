""" network structure for HSI
"""

import keras
from keras.layers import Conv2D, Input
from keras.models import Model

import keras_ipi

n_in = 132
n_code = 10


def gen_layer_in(h = 1, w = 1):
    shape = (h, w, n_in)
    return Input(shape = shape, name = 'input_x')


def gen_layer_code():
    shape = (1, 1, n_code )
    return Input(shape = shape, name = 'input_code')
    

def gen_encoder():
    layer_in = gen_layer_in()
    act_reg1 = keras_ipi.regularizers.Max1_reg(l = 0)
    act_reg2 = keras_ipi.regularizers.ClassBal_reg(l = 0.01)
    act_reg = keras_ipi.regularizers.Multi_reg(act_reg1, act_reg2)
    # act_reg = None  # TODO
    layer_encode = Conv2D(n_code, (3, 3), padding='same', activation='softmax', activity_regularizer=act_reg)(layer_in)
    model = Model(layer_in, layer_encode, name = 'encoder')
    
    return model


def gen_decoder():
    layer_code = gen_layer_code()
    layer_decode = Conv2D(n_in, (3, 3), activation='linear', padding='same')(layer_code)
    model = Model(layer_code, layer_decode, name = 'decoder')
    
    return model


class AutoEncoder():
    folder_model = '/home/lameeus/data/hsi/network_weights/'
    def __init__(self):
        """ the auto encoder network
        """
        
        w = 10
        layer_in = gen_layer_in(w, w)
        self.encoder = gen_encoder()
        self.decoder = gen_decoder()
        
        auto = self.decoder(self.encoder(layer_in))
        
        self.model = Model(layer_in, auto)
        self.model_encoder = Model(layer_in, self.encoder(layer_in))

        loss = keras.losses.mean_squared_error
        psnr = keras_ipi.metrics.psnr
        metrics = [loss, psnr]
        
        optimizer = {'class_name': 'adam', 'config': {'lr': 1.0e-4, 'beta_1': 0.90}}  # otherwise  = 'adam'
        self.model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics
                      )
        
    def stop(self):
        # TODO, release all GPU resources
        # TODO seems to work!
        import gc
        del self.model
        for i in range(15):
            gc.collect()

    def load(self):
        name = 'w_encoder.h5'
        self.encoder.load_weights(self.folder_model + name)
        name = 'w_decoder.h5'
        self.decoder.load_weights(self.folder_model + name)
    
    def save(self):
        name = 'w_encoder.h5'
        self.encoder.save_weights(self.folder_model + name)
        name = 'w_decoder.h5'
        self.decoder.save_weights(self.folder_model + name)
        
    def train(self, x, epochs = 1, save = True):
        """ no need of y since in auto encoder output is input """
        for i in range(epochs):
            print('Epoch {}/{}'.format(i+1, epochs))
            self.model.fit(x, x, epochs= 1)
            if save == True:
                self.save()
        
    def predict(self, x):
        return self.model.predict(x)
    
    def predict_code(self, x):
        return self.model_encoder.predict(x)
