""" network structure for HSI
"""

import keras
from keras.layers import Conv2D, Input
from keras.models import Model

import link_to_keras_ipi as keras_ipi

n_in = 132
n_code = 10
n_out = 8
l = 1e-5


def gen_layer_in(w = 1, ext = 0, nr = None):
    shape = (w + 2*ext, w + 2*ext, n_in)
    if nr is None:
        name = 'input_x'
    else:
        name = 'input_x_{}'.format(nr)
    return Input(shape = shape, name = name)


def gen_layer_code(w, ext):
    shape = (w + 2*ext, w + 2*ext, n_code )
    return Input(shape = shape, name = 'input_code')
    

def gen_encoder(w, ext):
    layer_in = gen_layer_in(w, ext)
    # act_reg1 = keras_ipi.regularizers.Max1_reg(l = 0)
    # act_reg2 = keras_ipi.regularizers.ClassBal_reg(l = 0.01)
    # act_reg = keras_ipi.regularizers.Multi_reg(act_reg1, act_reg2)
    # act_reg = None  # TODO
    
    l2 = keras.regularizers.l2(l)
    
    layer_encode = Conv2D(2*n_code, (2, 2), padding='valid', activation='elu', kernel_regularizer=l2)(layer_in)
    layer_encode = Conv2D(n_code, (2, 2), padding='valid', activation='softmax', kernel_regularizer=l2)(layer_encode)
    model = Model(layer_in, layer_encode, name = 'encoder')
    
    return model


def gen_decoder(w, ext):
    l2 = keras.regularizers.l2(l)
    
    layer_code = gen_layer_code(w, ext)
    layer_decode = Conv2D(2*n_code, (2, 2), activation='elu', padding='valid', kernel_regularizer=l2)(layer_code)
    layer_decode = Conv2D(n_in, (2, 2), activation='linear', padding='valid', kernel_regularizer=l2)(layer_decode)
    model = Model(layer_code, layer_decode, name = 'decoder')
    
    return model


def gen_classifier(w, ext):
    l2 = keras.regularizers.l2(l)
    
    layer_code = gen_layer_code(w, ext)
    layer_classifier = Conv2D(2*n_code, (2, 2), activation='elu', padding='valid', kernel_regularizer=l2)(layer_code)
    layer_classifier = Conv2D(n_out, (2, 2), activation='softmax', padding='valid', kernel_regularizer=l2)(layer_classifier)
    model = Model(layer_code, layer_classifier, name='classifier')
    
    return model


class Network():
    folder_model = '/home/lameeus/data/hsi/network_weights/'
    
    ext = 2
    w = 10
    
    def __init__(self):
        """ the auto encoder network
        """
        
        w = self.w
        ext = self.ext
        layer_in_auto = gen_layer_in(w, ext, nr = 0)
        layer_in_discr = gen_layer_in(w, ext, nr = 1)
        self.encoder = gen_encoder(w, ext)
        self.decoder = gen_decoder(w, ext // 2)
        self.classifier = gen_classifier(w, ext // 2)
        
        auto = self.decoder(self.encoder(layer_in_auto))
        discr = self.classifier(self.encoder(layer_in_discr))
        
        self.model_auto = Model(layer_in_auto, auto)
        self.model_encoder = Model(layer_in_discr, self.encoder(layer_in_discr))
        self.model_discr = Model(layer_in_discr, discr)
        self.model_all = Model([layer_in_auto, layer_in_discr], [auto, discr])
        
        loss = keras.losses.mean_squared_error
        psnr = keras_ipi.metrics.psnr
        metrics = [loss, psnr]
        
        optimizer = {'class_name': 'adam', 'config': {'lr': 1.0e-4, 'beta_1': 0.90}}  # otherwise  = 'adam'
        self.model_auto.compile(loss=loss,
                                optimizer=optimizer,
                                metrics=metrics
                                )
        
        loss_cross = keras.losses.categorical_crossentropy
        self.model_discr.compile(loss=loss_cross,
                           optimizer=optimizer,
                           # metrics=metrics
                           )

        self.model_all.compile(loss=[loss, loss_cross],
                                 optimizer=optimizer,
                                 # metrics=metrics
                                 )
    
    def stop(self):
        # TODO, release all GPU resources
        # TODO seems to work!
        import gc
        del self.model_auto
        for i in range(15):
            gc.collect()
    
    def load(self):
        name = 'w_encoder.h5'
        self.encoder.load_weights(self.folder_model + name)
        name = 'w_decoder.h5'
        self.decoder.load_weights(self.folder_model + name)
        name = 'w_discr.h5'
        self.classifier.load_weights(self.folder_model + name)
    
    def save(self):
        name = 'w_encoder.h5'
        self.encoder.save_weights(self.folder_model + name)
        name = 'w_decoder.h5'
        self.decoder.save_weights(self.folder_model + name)
        name = 'w_discr.h5'
        self.classifier.save_weights(self.folder_model + name)
        
    def train_auto(self, x, epochs = 1, save = True):
        """ no need of y since in auto encoder output is input """

        ext = self.ext
        x_out = x[:, ext:-ext, ext:-ext, :]
        for i in range(epochs):
            print('Epoch {}/{}'.format(i+1, epochs))
            self.model_auto.fit(x, x_out, epochs= 1)
            if save == True:
                self.save()

    def train_discr(self, x, y, epochs=1, save=True):
        """ no need of y since in auto encoder output is input """
    
        for i in range(epochs):
            print('Epoch {}/{}'.format(i + 1, epochs))
            self.model_discr.fit(x, y, epochs=1)
            if save == True:
                self.save()
                
    def train_all(self, x, y, x_auto = None, epochs=1, save=True):
        """ no need of y since in auto encoder output is input """
        
        if x_auto is None:
            x_placeholder = x
        else:
            x_placeholder = x_auto
            
        ext = self.ext
        x_out = x_placeholder[:, ext:-ext, ext:-ext, :]
        for i in range(epochs):
            print('Epoch {}/{}'.format(i + 1, epochs))
            self.model_all.fit([x_placeholder, x], [x_out, y], epochs=1)
            if save == True:
                self.save()

    def predict_auto(self, x):
        return self.model_auto.predict(x)

    def predict_discr(self, x):
        return self.model_discr.predict(x)

    def predict_code(self, x):
        return self.model_encoder.predict(x)
