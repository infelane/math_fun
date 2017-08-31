""" the MAIN thingy of July """

import keras
import numpy as np

import keras_ipi
import lambnet

# folder_loc = '/ipi/private/lameeus/private_Documents/python/2017_06'


def load_xy(set):
    """ COPIED FROM main_art_meeting """
    folder = '/home/lameeus/data/ghent_altar/input_arrays/'
    
    if set == 'hand':
        xy = np.load(folder + 'xy_hand_ext7.npz')

    return xy['x'], xy['y']


def gen_model():
    input_img = gen_layer_in()
    foo = keras_ipi.layers.Cropping2D(((5, 5), (5, 5)))(input_img)
    foo = keras.layers.Conv2D(2, (5,5), activation= 'sigmoid'
                              )(foo)
    foo = gen_layer_end()(foo)
    
    single = keras_ipi.layers.Cropping2D(((4, 4), (4, 4)))(input_img)
    single = keras.layers.Conv2D(1, (1,1), activation='sigmoid')(single)
    single = keras.layers.Conv2D(1, (7,7), activation='elu')(single)
    single = gen_layer_end()(single)
    
    model = keras.models.Model(input_img, foo)
    model_single = keras.models.Model(input_img, single)
    
    compile(model_single)
    
    return model_single


def gen_layer_end():
    depth_out = 2
    return keras.layers.Conv2D(depth_out, (1, 1), activation='softmax')


def gen_layer_in():
    ext = 7
    w = 7
    depth_in = 7
    shape_in = (w + 2 * ext, w + 2 * ext, depth_in)
    layer_in = keras.layers.Input(shape=shape_in)
    return layer_in


def layer_inception(layer_input):
    tower_1 = keras.layers.Conv2D(64, (1, 1), padding='same', activation='relu')(layer_input)
    tower_1 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = keras.layers.Conv2D(64, (1, 1), padding='same', activation='relu')(layer_input)
    tower_2 = keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(layer_input)
    tower_3 = keras.layers.Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

    output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=-1)
    return output


def compile(model):
    dice = keras_ipi.metrics.dice
    metrics = [dice]
    loss = keras_ipi.losses.bin_square()
    optimizer = {'class_name': 'adam', 'config': {'lr': 1.0e-3, 'beta_1': 0.90}} #otherwise  = 'adam'
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics
                  )


def gen_model2():
    layer_in = gen_layer_in()

    encoding_dim = 12

    layer_encode = keras_ipi.layers.Cropping2D(((1, 1), (1, 1)))(layer_in)
    # layer_encode = keras.layers.Conv2D(encoding_dim, (3,3), activation='sigmoid',
    #                         )(layer_encode)
    
    layer_encode = layer_inception(layer_encode)    #layer_encode
    layer_encode = keras_ipi.layers.Cropping2D(((6, 6), (6, 6)))(layer_encode)
    
    # layer_decoder = keras.la

    model_encoder = keras.models.Model(layer_in, layer_encode)
    
    layer_last = gen_layer_end()(model_encoder.output)
    
    model_ff = keras.models.Model(layer_in, layer_last)
    compile(model_ff)
    return model_ff


def gen_model3():
    layer_in = gen_layer_in()
    
    w_smallest = 1
    act1 = 'elu'
    
    from keras.layers import Conv2D, MaxPooling2D, concatenate, Conv2DTranspose
    
    in_crop = keras_ipi.layers.Cropping2D(((6, 7), (6, 7)))(layer_in)
    
    

    conv1 = Conv2D(32, (3, 3), activation=act1, padding='same')(in_crop)
    conv1 = Conv2D(32, (3, 3), activation=act1, padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation=act1, padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation=act1, padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation=act1, padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation=act1, padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    def foo(layer, width):
        """ the width of a layer """
        width_layer = (layer._keras_shape)[1]
        if width_layer != width:
            print(layer._name)
            print(width_layer)
            print(width)
            
            raise ValueError

    foo(pool3, w_smallest)
    foo(pool2, w_smallest*2)
    foo(pool1, w_smallest*4)
    foo(in_crop, w_smallest*8)

    # print(layer_in._keras_shape)
    # print(pool1._keras_shape)
    # print(pool2._keras_shape)
    # print(pool3._keras_shape)

    # conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    # conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    


    # up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation=act1, padding='same')(pool3)
    conv6 = Conv2D(256, (3, 3), activation=act1, padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation=act1, padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation=act1, padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation=act1, padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation=act1, padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation=act1, padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation=act1, padding='same')(conv9)

    conv10 = Conv2D(2, (1, 1), activation='softmax')(conv9)
    
    end_crop = keras_ipi.layers.Cropping2D(((1, 0), (1, 0)))(conv10)

    model = keras.models.Model(inputs=[layer_in], outputs=[end_crop])

    smooth = 1.
    def dice_coef(y_true, y_pred):
        import keras.backend as K
        y_true_f = y_true[..., 1]   # K.flatten(y_true)
        y_pred_f = y_pred[..., 1]   # K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_coef_loss(y_true, y_pred):
        return -dice_coef(y_true, y_pred)
    
    dice_ipi = keras_ipi.metrics.dice
    
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef, dice_ipi])
    return model

def train_net():
    x, y = load_xy('hand')
    
    n_total = len(x)
    print(n_total)
    n_train = int(n_total*0.8)
    x_train = x[:n_train, ...]
    y_train = y[:n_train, ...]
    x_test = x[n_train:, ...]
    y_test = y[n_train:, ...]
    print(np.shape(x))
    print(np.shape(y))

    model = gen_model3()
    print(model.summary())

    filepath = '/home/lameeus/data/ghent_altar/net_weight/2017_07/v_unet.h5'
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, verbose=0,
                                    save_weights_only=True, period=1)
        
    callbacks_list = [checkpoint]
    if 1:
        load_weight(model)

    model.fit(x_train, y_train,
              batch_size=32, epochs=10, shuffle=True,
              verbose=1,  # how much information to show 1 much or 0, nothing
              # class_weight= (1.0, 10.0),
              validation_data=(x_test, y_test),
              callbacks=callbacks_list
              )
    

def load_weight(model):
    filepath = '/home/lameeus/data/ghent_altar/net_weight/2017_07/v_unet.h5'
    model.load_weights(filepath)
    
    
def net_test():
    width = 7
    ext = 7
    model = gen_model3()

    load_weight(model)
    
    info = lambnet.block_info.Info(model)
    # info.output_test(width, ext, set='zach_small')
    # info.output_vis(width, ext, set = 'hand', bool_save = False, last_layer = False)
    info.certainty(width, ext, set = 'hand')
    # info.certainty(width, ext, set = 'zach_small')
    # info.certainty(width, ext, set = 'zach')
    info.certainty(width, ext, set = 'hand_small')
    

def net_other():
    model = gen_model3()


def main():
    train_net()
    net_test()
    
    net_other()
    

if __name__ == '__main__':
    main()
    