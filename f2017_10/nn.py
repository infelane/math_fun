from keras.models import Model
from keras.layers import concatenate, MaxPool2D, Conv2D, Concatenate, Conv2DTranspose, UpSampling2D, AvgPool2D, Cropping2D
import keras

from link_to_keras_contrib_lameeus.keras_contrib.callbacks.dead_relu_detector import DeadReluDetector

import link_to_keras_ipi as keras_ipi
from f2017_08.hsi.tools_network import gen_in, BaseNetwork, relu, softmax, inception


class Network(BaseNetwork):
    def __init__(self, w = 10, zoom = 1, lr = 1e-4):
        # w = 10
        ext = 3
        ext_zoom = zoom *ext
        
        layer_clean = gen_in(w, ext_zoom, 3, name = 'in_clean')
        layer_rgb = gen_in(w, ext_zoom, 3, name='in_rgb')
        layer_ir = gen_in(w, ext_zoom, 1, name='in_ir')
        inputs = [layer_clean, layer_rgb, layer_ir]
        
        layer_model = concatenate([layer_clean, layer_rgb, layer_ir])
        layer_inc = inception(layer_model, 50, zoom = zoom, activation='relu')
        layer_inc = Cropping2D(((zoom, zoom), (zoom, zoom)))(layer_inc)
        
        unet1 = MaxPool2D(pool_size=(2*zoom, 2*zoom), strides=(1, 1))(layer_model)
        unet1 = Conv2D(100, (3, 3), dilation_rate=2*zoom)(unet1)   # dilation rate because of pooling
        # unet10 = Conv2DTranspose(50, (2, 2), strides=(1, 1), dilation_rate=1)(te_unet) # again increase size by 1
        # unet10 = Conv2DTranspose(50, (2, 2), strides=(2, 2), dilation_rate=1)(te_unet)  # again increase size by 1
        # unet10 = UpSampling2D(size = (2, 2), dil)(te_unet)
        #
        # unet10 = Conv2DTranspose(50, (1, 1))(te_unet)  # again increase size by 1
        # unet10 = AvgPool2D((2*zoom, 2*zoom), strides =(1, 1))(unet10)
        
        # Correct one
        unet10 = Conv2D(50, (2, 2))(unet1)
        
        layer_model = Concatenate()([layer_inc, unet10])
        
        layer_model = relu(10, (1, 1), zoom)(layer_model)
        outputs = softmax(2, (1, 1), zoom)(layer_model)
        
        dice = keras_ipi.metrics.dice_with_0_labels
        metrics = [dice]
        loss = keras.losses.categorical_crossentropy
        optimizer = keras.optimizers.adam(lr = lr, clipnorm = 1.)
 
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=metrics
                           )

        file_name = 'w_stacker_unet'
        folder_model = '/home/lameeus/data/ghent_altar/net_weight/2017_10/'

        super().__init__(file_name, self.model, folder_model)

        # cb_saver = keras.callbacks.LambdaCallback(on_epoch_end=lambda *a: self.save())
        # self.callback.append(cb_saver)

        # self.callback.append(DeadReluDetector)



    def train(self, x, y, epochs=1, save=True, validation_split=None, verbose=1):
        
        x_crop = [x_i[0:10,:,:,:] for x_i in x]
        
        # cb_saver = keras.callbacks.LambdaCallback(on_epoch_end=lambda *a: self.save())
        # self.callback.append(cb_saver)
        
        cb = DeadReluDetector(x_train=x, verbose=True)
        self.callback.append(cb)

        super().train(x, y, epochs=epochs, save=save, validation_split=validation_split, verbose=verbose)
