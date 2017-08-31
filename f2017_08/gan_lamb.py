import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D, concatenate
from keras.losses import mean_squared_error
from keras.models import Model

import keras_ipi
import lambnet
from maus.paint_tools import image_tools


def gen_encoder(layer_in):
    x = Conv2D(5, (1,1), activation='tanh', padding='valid')(layer_in)
    return Model(layer_in, x)
  
    
def gen_decoder(layer_code):
    x = Conv2DTranspose(3, (1, 1), activation = 'tanh', padding = 'valid')(layer_code)
    return Model(layer_code ,x)


def show_images(im1, im2):
    plt.subplot(2, 1, 1)
    plt.imshow(im1)
    plt.subplot(2, 1, 2)
    plt.imshow(im2)
    plt.show()
    
    
def gen_auto():
    width = 10
    
    shape_input = (width, width, 3)
    
    layer_in1 = Input(shape_input, name='x_in')
    layer_in2 = Input(shape_input, name='x_in2')

    star_1 = encoder1(layer_in1)
    star_2 = encoder2(layer_in2)

    x_11 = decoder1(star_1)
    x_22 = decoder2(encoder2(layer_in2))
    x_121 = decoder1(encoder2(decoder2(star_1)))
    x_212 = decoder2(encoder1(decoder1(star_2)))
    
    inputs = [layer_in1, layer_in2]
    outputs = [x_11, x_22, x_121, x_212]
    
    adam = keras.optimizers.adam(lr=1e-3)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=adam,
                  loss=mean_squared_error
                  )
    
    return model
    
def part_auto_encoder():
    path = '/home/lameeus/data/ghent_altar/input/19_clean.tif'
    im1 = image_tools.path2im(path)
    
    path = '/home/lameeus/data/ghent_altar/input/13_new_clean_reg1.tif'
    im2 = image_tools.path2im(path)
    
    if 0:
        show_images(im1, im2)
    
    width = 10

    shape_code = (width, width, 5)

    layer_code1 = Input(shape_code, name='x_code')
    layer_code2 = Input(shape_code, name='x_code2')
    
    layer_in1_all = Input(np.shape(im1), name='x_im1')
    layer_in2_all = Input(np.shape(im2), name='x_im2')
    
    encoder1 = gen_encoder(layer_in1)
    encoder2 = gen_encoder(layer_in2)
    decoder1 = gen_decoder(layer_code1)
    decoder2 = gen_decoder(layer_code2)
    
    x_11_all = decoder1(encoder1(layer_in1_all))
    x_12_all = decoder2(encoder1(layer_in1_all))
    x_121_all = decoder1(encoder2(x_12_all))
    x_22_all = decoder2(encoder2(layer_in2_all))
    x_21_all = decoder1(encoder2(layer_in2_all))
    x_212_all = decoder2(encoder1(x_21_all))
    
    model = gen_auto()
    
    model1 = Model(inputs=layer_in1_all, outputs=[x_11_all, x_12_all, x_121_all])
    model2 = Model(inputs=layer_in2_all, outputs=[x_22_all, x_21_all, x_212_all])
    
    shape = np.shape(im1)
    shape2 = np.shape(im2)
    n_train = 2000
    h_random = np.random.randint(0, shape[0] - width, (n_train,))
    w_random = np.random.randint(0, shape[1] - width, (n_train,))
    h_random2 = np.random.randint(0, shape2[0] - width, (n_train,))
    w_random2 = np.random.randint(0, shape2[1] - width, (n_train,))
    
    x_train1 = []
    x_train2 = []
    for i in range(n_train):
        h_0 = h_random[i]
        h_1 = h_0 + width
        w_0 = w_random[i]
        w_1 = w_0 + width
        
        h_2 = h_random2[i]
        w_2 = w_random2[i]
        
        print(h_0)
        
        x_train1.append(im1[h_0:h_1, w_0:w_1, ...])
        x_train2.append(im2[h_2:h_2 + width, w_2:w_2 + width, ...])
    
    x_train1 = np.array(x_train1)
    x_train2 = np.array(x_train2)
    
    def save_submodels():
        folder_model = '/home/lameeus/data/ghent_altar/net_weight/gan_lamb/'
        name = 'encoder1.h5'
        encoder1.save_weights(folder_model + name)
        name = 'encoder2.h5'
        encoder2.save_weights(folder_model + name)
        name = 'decoder1.h5'
        decoder1.save_weights(folder_model + name)
        name = 'decoder2.h5'
        decoder2.save_weights(folder_model + name)
    
    def load_submodels():
        folder_model = '/home/lameeus/data/ghent_altar/net_weight/gan_lamb/'
        name = 'encoder1.h5'
        encoder1.load_weights(folder_model + name)
        name = 'encoder2.h5'
        encoder2.load_weights(folder_model + name)
        name = 'decoder1.h5'
        decoder1.load_weights(folder_model + name)
        name = 'decoder2.h5'
        decoder2.load_weights(folder_model + name)
    
    if 1:
        load_submodels()
    
    n_epochs = 100
    
    if n_epochs:
        model.fit([x_train1, x_train2], [x_train1, x_train2, x_train1, x_train2], epochs=n_epochs)
        save_submodels()
        
    if 0:   # plot the auto-encoder results
        x_im1 = np.stack([im1], axis=0)
        x_im2 = np.stack([im2], axis=0)
        
        y_im1 = model1.predict(x_im1)
        y_im2 = model2.predict(x_im2)
        
        plt.subplot(2, 3, 1)
        plt.imshow(y_im1[0][0, ...])
        plt.subplot(2, 3, 2)
        plt.imshow(y_im1[1][0, ...])
        plt.subplot(2, 3, 3)
        plt.imshow(y_im1[2][0, ...])
        plt.subplot(2, 3, 4)
        plt.imshow(y_im2[0][0, ...])
        plt.subplot(2, 3, 5)
        plt.imshow(y_im2[1][0, ...])
        plt.subplot(2, 3, 6)
        plt.imshow(y_im2[2][0, ...])
        plt.show()


def gen_layer_in():
    ext = 7
    # w = 7
    w = 8
    depth_in = 7
    shape_in = (w + 2 * ext, w + 2 * ext, depth_in)
    layer_in = keras.layers.Input(shape=shape_in)
    return layer_in


def gen_model3():
    layer_in = gen_layer_in()
    
    w_smallest = 1
    act1 = 'elu'
    
    from keras.layers import Conv2D, MaxPooling2D, concatenate, Conv2DTranspose
    
    in_crop = keras_ipi.layers.Cropping2D(((7, 7), (7, 7)))(layer_in)
    
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
    foo(pool2, w_smallest * 2)
    foo(pool1, w_smallest * 4)
    foo(in_crop, w_smallest * 8)
    
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
    
    end_crop = keras_ipi.layers.Cropping2D(((0, 0), (0, 0)))(conv10)
    
    model = keras.models.Model(inputs=[layer_in], outputs=[end_crop])
    
    smooth = 1.
    
    def dice_coef(y_true, y_pred):
        import keras.backend as K
        y_true_f = y_true[..., 1]  # K.flatten(y_true)
        y_pred_f = y_pred[..., 1]  # K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
    def dice_coef_loss(y_true, y_pred):
        return -dice_coef(y_true, y_pred)
    
    dice_ipi = keras_ipi.metrics.dice
    
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef, dice_ipi])
    return model


def gen_discr(w, ext, padding = 'valid'):
#     ext = 7
#     w = 7
    
    shape_x = (w + 2 * ext, w + 2 * ext, 7)
    shape_y = (w, w, 2)
    
    input_x = Input(shape=shape_x)
    input_y = Input(shape=shape_y)
    
    crop_x = keras.layers.Cropping2D(((ext, ext), (ext, ext)))(input_x)
    crop_y = keras_ipi.layers.CroppDepth((0, 1))(input_y)
    
    discr = keras.layers.concatenate([crop_x, crop_y], axis=3)
    # discr = keras_ipi.layers.Cropping2D(((3, 3), (3, 3)))(discr)
    # discr = Conv2D(10, (3, 3), activation='elu', padding=padding)(discr)
    # discr = Conv2D(3, (3, 3), activation='softmax', padding=padding)(discr)






    layer_in = gen_layer_in()
    
    w_smallest = 1
    act1 = 'elu'
    
    
    
    in_crop = keras_ipi.layers.Cropping2D(((0, 0), (0, 0)))(discr)
    
    conv1 = Conv2D(32, (3, 3), activation=act1, padding='same')(in_crop)
    conv1 = Conv2D(32, (3, 3), activation=act1, padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, (3, 3), activation=act1, padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation=act1, padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, (3, 3), activation=act1, padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation=act1, padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(conv3)
    
    
    # def foo(layer, width):
    #     """ the width of a layer """
    #     width_layer = (layer._keras_shape)[1]
    #     if width_layer != width:
    #         print(layer._name)
    #         print(width_layer)
    #         print(width)
    #
    #         raise ValueError
    #
    #
    # foo(pool3, w_smallest)
    # foo(pool2, w_smallest * 2)
    # foo(pool1, w_smallest * 4)
    # foo(in_crop, w_smallest * 8)
    
    # print(layer_in._keras_shape)
    # print(pool1._keras_shape)
    # print(pool2._keras_shape)
    # print(pool3._keras_shape)
    
    # conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    # conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    
    
    
    # up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation=act1, padding='same', dilation_rate=(2,2), strides=(1,1))(pool3)
    conv6 = Conv2D(256, (3, 3), activation=act1, padding='same', dilation_rate=(2,2), strides=(1,1))(conv6)
    
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(1, 1), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation=act1, padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation=act1, padding='same')(conv7)
    
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation=act1, padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation=act1, padding='same')(conv8)
    
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation=act1, padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation=act1, padding='same')(conv9)
    
    conv10 = Conv2D(3, (1, 1), activation='softmax')(conv9)
    
    end_crop = keras_ipi.layers.Cropping2D(((0, 0), (0, 0)))(conv10)
    
    # model = keras.models.Model(inputs=[layer_in], outputs=[end_crop])



   
    # model_discr = Model(inputs=[input_x, input_y], outputs=[discr])
    model_discr = Model(inputs=[input_x, input_y], outputs=[end_crop])
    
    model_segm = gen_model3()
    filepath = '/home/lameeus/data/ghent_altar/net_weight/2017_07/v_unet.h5'
    model_segm.load_weights(filepath)
    
    model_segm.trainable = False
    
    import tensorflow as tf
    # round off such that the network can't distinguish based on integer vs float
    rounder = keras.layers.Lambda(lambda x : tf.round(x))
    y_pred_round = rounder(model_segm(input_x))
    
    d_true = model_discr([input_x, input_y])
    d_pred = model_discr([input_x, y_pred_round])
    
    # model_discr.compile(optimizer=keras.optimizers.adam(lr=1e-4), loss=keras.losses.categorical_crossentropy)
    # return model_discr
    model = Model([input_x, input_y], [d_true, d_pred])
    model.compile(optimizer=keras.optimizers.adam(lr=1e-5), loss = keras.losses.categorical_crossentropy)
    return model, model_discr


def part_discriminator():
    # w = 7
    w = 8     # 103, 503, 7 # a number that is MOD 8 = 7 or w + 1 is divisible by 8
    ext = 7
    
    model, model_discr = gen_discr(w, ext, padding = 'same')
    
    w = 200    # 103, 503, 7 # a number that is MOD 8 = 7 or w + 1 is divisible by 8
    ext = 7
    model_same, model_discr_same = gen_discr(w, ext, padding='same')

    path_discr = '/home/lameeus/data/ghent_altar/net_weight/lamb_discr/a.h5'
    
    if 1:
        model_discr.load_weights(path_discr)
        
    if 0:
        """ COPIED FROM main_art_meeting """
        folder = '/home/lameeus/data/ghent_altar/input_arrays/'
    
        set = 'hand'
        if set == 'hand':
            xy = np.load(folder + 'xy_hand_ext8.npz')
    
        x = xy['x']
        y = xy['y']
        
        print(np.shape(x))
        print(np.shape(y))
        #
        # 1/0
    
        # y_discr_true = np.zeros((len(x), 3, 3, 3))
        # y_discr_pred = np.zeros((len(x), 3, 3, 3))
        # y_discr_true[..., 0:2] = y[:, 2:5, 2:5, 0:2]
        
        y_discr_true = np.zeros((len(x), 8, 8, 3))
        y_discr_pred = np.zeros((len(x), 8, 8, 3))
        y_discr_true[..., 0:2] = y[:, :, :, 0:2]
        y_discr_pred[..., 2] = 1
        
        model.fit([x, y], [y_discr_true, y_discr_pred], epochs=50000)
        model_discr.save_weights(path_discr)
    
    # foo = model.predict([x,y])
    # print(foo)

    model_discr_same.load_weights(path_discr)
    info = lambnet.block_info.Info(model_same)
    # info.output_test(width, ext, set='zach_small')
    # info.output_vis(width, ext, set = 'hand', bool_save = False, last_layer = False)
    # info.certainty(7, 7, set='hand')
    # info.certainty(width, ext, set = 'zach_small')
    # info.certainty(7, 7, set='hand_small')
    
    # info.certainty_discr(7, 7, set='hand_small')
    info.certainty_discr(w, ext, set='hand')


def part_testing():
    width = 7
    ext = 7
    model = gen_model3()
    
    filepath = '/home/lameeus/data/ghent_altar/net_weight/2017_07/v_unet.h5'
    model.load_weights(filepath)
    
    info = lambnet.block_info.Info(model)
    # info.output_test(width, ext, set='zach_small')
    # info.output_vis(width, ext, set = 'hand', bool_save = False, last_layer = False)
    info.certainty(width, ext, set='hand')
    # info.certainty(width, ext, set = 'zach_small')
    # info.certainty(width, ext, set = 'zach')
    info.certainty(width, ext, set='hand_small')

def main():
    if 0:
        part_auto_encoder()

        part_testing()
        
    part_discriminator()

    
if __name__ == '__main__':
    main()