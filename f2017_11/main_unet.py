"""
Copy unet architecture
"""

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Add, Multiply, Lambda, Cropping2D, AveragePooling2D
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.regularizers import l2
import keras.backend as K
import numpy as np
import keras.callbacks
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

from f2017_08.hsi import tools_data, tools_plot
from f2017_08.hsi.tools_network import gen_in, BaseNetwork, relu, softmax, inception
from f2017_09 import main_lamb
import link_to_keras_ipi as keras_ipi
from link_to_keras_ipi.layers import CroppDepth
from link_to_keras_contrib_lameeus.keras_contrib.callbacks.dead_relu_detector import DeadReluDetector
from link_to_keras_ipi.preprocessing.image import ImageDataGenerator as ImageDataGenerator2
from link_to_soliton.paint_tools import image_tools

# reduce GPU usage to 80%
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.85   # From 0.9
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))

# img_rows = 96
# img_cols = 96

# For training, to limit memory
img_rows = 50
img_cols = 50

# img_rows = 60
# img_cols = 60

# img_rows = 20
# img_cols = 20

# img_rows = 108  # height is good
# img_cols = 108

# img_rows = 120
# img_cols = 120

# img_rows = 64   # quite good
# img_cols = 64

# img_rows = 100  # height is good
# img_cols = 108  # width is good

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def model_adop():
    # TODO get automatic
    n_pad = 2
    
    in_clean = Input((img_rows + n_pad, img_cols + n_pad, 3))

    # TODO change to
    paint_loss_input = Input((img_rows + n_pad, img_cols + n_pad, 1))

    padding = 'same'    # valid

    conv = Conv2D(100, (3, 3), activation='relu', padding=padding)(in_clean)
    conv = Conv2D(3, (3, 3), activation='linear', padding=padding)(conv)

    preproces = Lambda(lambda x : K.cast(K.greater_equal(x, 0.5), np.float32))(paint_loss_input)

    preproces_back = Lambda(lambda x : K.cast(K.greater_equal(0.5, x), np.float32))(paint_loss_input)

    mult = Multiply()([conv, preproces])
    mult_back = Multiply()([in_clean, preproces_back])

    # add = Add()([in_clean, mult])   # residue does not seem to work
    add = Add()([mult, mult_back])
    model = Model([in_clean, paint_loss_input], add)

    #TODO
    return model


def model_adop_stack():
    model_adop_start = model_adop()
    
    # TODO get automatic
    n_pad = 2

    in_clean = Input((img_rows + n_pad, img_cols + n_pad, 3))
    in_rgb = Input((img_rows + n_pad, img_cols + n_pad, 3))
    in_ir = Input((img_rows + n_pad, img_cols + n_pad, 1))

    unet = Network(1)
    model_unet = unet.model
    model_unet.trainable = False

    pred = model_unet([in_clean, in_rgb, in_ir])
    pred = CroppDepth(depth=(1, 2))(pred)
    
    # in_clean = model_adop_start.input[0]
    inputs = [in_clean, in_rgb, in_ir]
    outputs = model_unet([model_adop_start([in_clean, pred]), in_rgb, in_ir])
    model = Model(inputs = inputs, outputs=outputs)
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')

    model_save = model_adop_start
    model_adop_predict = Model(inputs, model_adop_start([in_clean, pred]))
    
    return model, model_save, model_adop_predict, unet


class NetworkAdop(BaseNetwork):
    def __init__(self):
        file_name = 'w_0'
        folder_model = '/home/lameeus/data/ghent_altar/net_weight/2017_11/inpainting_net/'

        model, model_save, model_adop_predict, unet = model_adop_stack()
        self.model_both = model
        self.model_save = model_save
        self.model_adop_predict = model_adop_predict
        self.unet = unet
        
        super().__init__(file_name, 1, folder_model)    # no need to set model

    def load(self, name=None, make_backup=False, also_adop=True):
        if name is None:
            name = self.file_name

        if also_adop:
            self.model_save.load_weights(self.folder_model + name + '.h5')
        
        self.unet.load()

        if make_backup:
            name_old = name + '_previous'
            self.model_save.save_weights(self.folder_model + name_old + '.h5')

    def save(self, name=None):
        if name is None:
            name = self.file_name

        self.model_save.save_weights(self.folder_model + name + '.h5')

    def train(self, x, y, epochs=1, save=True, validation_split=None, verbose=1):
        callback = self.callback

        if save:
            cb_saver = keras.callbacks.LambdaCallback(on_epoch_end=lambda *a: self.save())
            callback.append(cb_saver)
    
        # Adjusted to train on stacked network
        self.model_both.fit(x, y, epochs=epochs, validation_split=validation_split,
                       callbacks=callback, verbose=verbose)

    def predict(self, x):
        return self.model_adop_predict.predict(x)


def get_model_unet(in_list):
    depth_1 = 32
    
    conv1 = Conv2D(depth_1, (3, 3), activation='relu', padding='same')(in_list)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(depth_1 * 2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(depth_1 * 2, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(depth_1 * 4, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(depth_1 * 4, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(depth_1 * 8, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(depth_1 * 8, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(depth_1 * 16, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(depth_1 * 16, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(depth_1 * 8, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(depth_1 * 8, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(depth_1 * 4, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(depth_1 * 4, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(depth_1 * 2, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(depth_1 * 2, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(depth_1, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(depth_1, (3, 3), activation='relu', padding='same')(conv9)
    
    return conv9


def get_model_unet_v2(in_list):
    doubling = False     # to double or not the downpooled layers depth (False for memory usage restrictions)
    
    padding = 'valid'   # same or valid

    kernel_regularizer = l2(1.e-6)
    
    n_pad = 0

    n_pad_dict = {i : 0 for i in range(4)}
    
    depth_1 = 32
    
    if doubling == True:
        def depth_i(a):
            """
            :param a: dilation rate
            :return:
            """
            return depth_1 * a
        
    else:
        def depth_i(i):
            return depth_1 * 2

    #level 0, x1, left
    i = 0
    dilation_rate = 2 ** i
    conv1 = Conv2D(depth_i(dilation_rate), (3, 3), activation='relu', padding=padding,
                   kernel_regularizer=kernel_regularizer)(in_list)
    
    def down_part(i, conv_down_prev_i, n_pad, n_pad_dict, padding=padding):
        # TODO
        dilation_rate = 2 ** i
        
        # TODO switched to MeanPooling since it makes more sense, more shift invariant
        pool_i = AveragePooling2D(pool_size=(dilation_rate, dilation_rate), strides=1, padding=padding)(conv_down_prev_i)
        conv_down_i = Conv2D(depth_i(dilation_rate), (3, 3), dilation_rate=dilation_rate, activation='relu',
                             padding=padding, kernel_regularizer=kernel_regularizer)(pool_i)
        conv_down_i = Conv2D(depth_i(dilation_rate), (3, 3), dilation_rate=dilation_rate, activation='relu',
                             padding=padding, kernel_regularizer=kernel_regularizer)(conv_down_i)
        
        if padding == 'valid':
            n_pad_i = dilation_rate*5-1
            # n_pad+=n_pad_i
            
            for k in range(i):  # does not include i, which it shoudln't update!
                n_pad_dict[k] += n_pad_i
            
        return conv_down_i, n_pad, n_pad_dict

    # level 1 x2, left
    conv2, n_pad, n_pad_dict = down_part(i=1, conv_down_prev_i=conv1, n_pad=n_pad, n_pad_dict=n_pad_dict)

    # level 2 x4, left
    conv3, n_pad, n_pad_dict = down_part(i=2, conv_down_prev_i=conv2, n_pad=n_pad, n_pad_dict=n_pad_dict)
    
    # #level 3 x8, left
    conv4, n_pad, n_pad_dict = down_part(i=3, conv_down_prev_i=conv3, n_pad=n_pad, n_pad_dict=n_pad_dict)  # , padding='same')

    #level 4, x16, Lowest level
    conv5, n_pad, n_pad_dict = down_part(i=4, conv_down_prev_i=conv4, n_pad=n_pad, n_pad_dict=n_pad_dict, padding='same') #, padding='same')
    
    def up_part_i(i, conv_up_prev_i, conv_short_i, n_pad, n_pad_dict, padding=padding):
        dilation_rate = 2 ** i

        if padding == 'valid':
            n_pad_prev = 1*dilation_rate + n_pad_dict[i] + n_pad
            
            ext_i = n_pad_prev//2
            ext_tuple_i = ((ext_i, n_pad_prev-ext_i), (ext_i, n_pad_prev-ext_i))
            crop_i = Cropping2D(ext_tuple_i)(conv_short_i)
        else:
            crop_i = conv_short_i
        
        #TODO fix!!
        depth_fix = depth_i(dilation_rate)  # 32*dilation_rate

        up_conv_i = concatenate([Conv2D(depth_fix, (2, 2), strides=(1, 1), dilation_rate=(dilation_rate),
                                        padding=padding, name='up_lvl{}'.format(i+1),
                                        kernel_regularizer=kernel_regularizer
                                        )(conv_up_prev_i), crop_i])
        conv_up_i = Conv2D(depth_i(dilation_rate), (3, 3), strides=(1, 1), dilation_rate=(dilation_rate),
                           activation='relu', padding=padding, name='conv_up1_l{}'.format(i+1),
                           kernel_regularizer=kernel_regularizer)(up_conv_i)
        conv_up_i = Conv2D(depth_i(dilation_rate), (3, 3), strides=(1, 1), dilation_rate=(dilation_rate),
                           activation='relu', padding=padding, name='conv_up2_l{}'.format(i+1),
                           kernel_regularizer=kernel_regularizer)(conv_up_i)
        
        if padding == 'valid':
            n_pad+=dilation_rate*5
        
        return conv_up_i, n_pad
        
    # level 3, x8, right
    conv6, n_pad = up_part_i(i=3, conv_up_prev_i=conv5, conv_short_i=conv4, n_pad=n_pad, n_pad_dict=n_pad_dict)
    
    # level 2, x4, right
    conv7, n_pad = up_part_i(i=2, conv_up_prev_i=conv6, conv_short_i=conv3, n_pad=n_pad, n_pad_dict=n_pad_dict)

    # level 1, x2, right
    conv8, n_pad = up_part_i(i=1, conv_up_prev_i=conv7, conv_short_i=conv2, n_pad=n_pad, n_pad_dict=n_pad_dict)

    # level 0, x1, right
    conv9, n_pad = up_part_i(i=0, conv_up_prev_i=conv8, conv_short_i=conv1, n_pad=n_pad, n_pad_dict=n_pad_dict)
    
    print('n_pad should be: {}'.format(n_pad))
    
    return conv9


def get_unet(input_v):
    depth_1 = 32
    
    n_pad = 0
    if 1:
        n_pad = 1*5+2*5+4*5+8*5+8*5-1+4*5-1+2*5-1+1*4  # 16*5-1 # TODO alter last 8*5-1  #2*2*4 + 1*1*4  # ext = n_pad//2
    
    in_clean = Input((img_rows+n_pad, img_cols+n_pad, 3))
    in_rgb = Input((img_rows+n_pad, img_cols+n_pad, 3))
    in_ir = Input((img_rows+n_pad, img_cols+n_pad, 1))

    in_clean_nn = Conv2D(32, (3, 3), activation='relu', padding='same')(in_clean)
    in_clean_nn = Conv2D(3, (1, 1), activation='linear', padding='same')(in_clean_nn)
    in_clean_nn = keras.layers.Add()([in_clean, in_clean_nn])
    in_clean_upgraded = Input((img_rows+n_pad, img_cols+n_pad, 3))
    
    model_unet_clean = Model(inputs=in_clean, outputs=in_clean_nn)

    in_list = [in_clean_upgraded, in_rgb, in_ir]
    in_list_upgraded = [in_clean, in_rgb, in_ir]
    in_list_upgraded2 = [model_unet_clean(in_clean), in_rgb, in_ir]
    
    if 0:
        inputs = Input((img_rows+n_pad, img_cols+n_pad, 1))
        in_list = [inputs]
        conv0 = Conv2D(depth_1, (3, 3), activation='linear', padding='same')(inputs)
        
    elif input_v == 1:
        # stacking at input (7 input channels instead of 1 (see above))
        in_conc = concatenate(in_list)
        kernel_regularizer = l2(1.e-6)
        conv0 = Conv2D(depth_1, (3, 3), activation='relu', padding='valid',
                       kernel_regularizer=kernel_regularizer)(in_conc)
    
    elif input_v == 0:
        # converting the input channels to single
        in_conc = concatenate(in_list)
        in_sum = Conv2D(1, (1, 1), activation='relu', padding='same')(in_conc)
        conv0 = Conv2D(depth_1, (3, 3), activation='relu', padding='same')(in_sum)
    else:
        raise ValueError('unknown input_v: {}'.format(input_v))

    # in_list is already a list
    model_unet_1 = Model(inputs=in_list, outputs=conv0)

    # in_list_sub = Input((img_rows, img_cols, 32))
    in_list_sub = Input(tensor=conv0)
    # in_list_sub(conv1)

    # in_list_sub

    name = 'layer_cut'

    conv9 = get_model_unet_v2(in_list_sub)
    
    # in_list_sub is a list already
    model_unet_without_1_end = Model(inputs=in_list_sub, outputs=[conv9])

    model_unet_without_1_end.trainable = True

    in_list_end = Input(tensor=conv9)
    kernel_regularizer = l2(1.e-6)
    conv10 = Conv2D(2, (1, 1), activation='softmax', kernel_regularizer=kernel_regularizer)(in_list_end)

    # model_sub = Model(inputs=in_list_sub, outputs=[conv10])

    # outputs = [conv10]

    # outputs = conv10(conv0)

    # outputs = model_sub(conv0)

    model_unet_end = Model(inputs = [in_list_end], outputs=[conv10])

    # model_unet_end.trainable = False
    # model_unet_without_1_end.trainable = False
    
    if 0: # make up_lvl untrainable
        for layer_i in model_unet_without_1_end.layers:
            if 'up_lvl' not in layer_i.name:
                layer_i.trainable = False
    
    # model_unet_without_1_end.get_layer().trainable = True
    # model_unet_1.trainable = False

    if 1:   # Without preprocessing on first modality
        total = model_unet_end(model_unet_without_1_end(model_unet_1(in_list)))
    
        # model = Model(inputs=in_list, outputs=conv10)
        model = Model(inputs=in_list, outputs=total)
    
    else:   # With preprocessing on first modality
        total = model_unet_end(model_unet_without_1_end(model_unet_1(in_list_upgraded2)))
        model = Model(inputs=in_list_upgraded, outputs=total)
    
    # model.trainable = False
    
    if 0:
        model.summary()
    if 1:
        model_unet_without_1_end.summary()

    # model.get_layer(name = name)

    dice = keras_ipi.metrics.dice_with_0_labels

    model.compile(optimizer=Adam(lr=1e-5), loss=categorical_crossentropy, metrics=[dice])

    return model, (model_unet_1, model_unet_without_1_end, model_unet_end, model_unet_clean), n_pad


class Network(BaseNetwork):
    def __init__(self, w=10, zoom=1, lr=1e-4):
        
        input_v = 1
        
        model, sub_models, n_pad = get_unet(input_v)
        folder_model = '/home/lameeus/data/ghent_altar/net_weight/2017_11/unet/'
        file_name = 'w_2'
        super().__init__(file_name, model, folder_model)
        self.sub_models = sub_models
        self.input_v = input_v
        self.n_pad = n_pad
        
    def train(self, x, y, epochs=1, save=True, validation_split=None, verbose=1):
        # x_crop = [x_i[0:10, :, :, :] for x_i in x]
        # cb = DeadReluDetector(x_train=x, verbose=True, bool_warning=False)
        # self.callback.append(cb)
        
        from keras.preprocessing.image import ImageDataGenerator
        
        if 0:
            datagen = ImageDataGenerator(horizontal_flip=True,
                                         vertical_flip=True,
                                         )
        
        else:
            # own version
            datagen = ImageDataGenerator2(horizontal_flip=True,
                                          vertical_flip=True,
                                          diagonal_flip=True
                                          )
        
        # super().train(x, y, epochs=epochs, save=save, validation_split=validation_split, verbose=verbose)

        callback = self.callback

        if save:
            cb_saver = keras.callbacks.LambdaCallback(on_epoch_end=lambda *a: self.save())
            callback.append(cb_saver)

        # self.model.fit(x, y, epochs=epochs, validation_split=validation_split,
        #                callbacks=callback, verbose=verbose)
        
        # TODO here

        batch_size = 16     # 32
        
        def generate_data_generator_list(x, y):
            seed = 1337
            assert type(x) == list
            
            n_in = len(x)

            gens_in = []
            
            for i in range(n_in):
                gens_in.append(datagen.flow(x[i], seed=seed, batch_size=batch_size))

            gens_out = datagen.flow(y, seed=seed, batch_size=batch_size)
            
            while True:
                xi = [a.next() for a in gens_in]
                yi = gens_out.next()
                
                yield xi, yi
        
        # train_generator = datagen.flow(x, y,
        #     # 'data/train',  # this is the target directory
        #     # target_size=(150, 150),  # all images will be resized to 150x150
        #     # batch_size=batch_size,
        #     # class_mode='binary'
        #     )  # since we use binary_crossentropy loss, we need binary labels

        n_data = np.shape(x[0])[0]  # x is a list (use x[0]). get the amount of samples
        

        self.model.fit_generator(generate_data_generator_list(x, y), epochs=epochs,
                                 # validation_split=validation_split,
                                 callbacks=callback,
                                 steps_per_epoch=n_data//batch_size,
                                 verbose=verbose)
    
    # def save_sub(self, name = None):
    #     if name is None:
    #         name = self.file_name
    #
    #     self.model.save_weights(self.folder_model + name + '_sub.h5')
    #
    # def load_sub(self, name=None):
    #
    #     if name is None:
    #         name = self.file_name
    #     self.model.load_weights(self.folder_model + name + '_sub.h5')
        
    def load(self, name=None):
        if name is None:
            name = self.file_name
            
        # for i, sub_model_i in enumerate(self.sub_models):
        #     sub_model_i.load_weights(self.folder_model + name + '_{}.h5'.format(i))
        for i, sub_model_i in enumerate(self.sub_models):
            if i == 0:
                sub_model_i.load_weights(self.folder_model + name + '_{}_v{}.h5'.format(i, self.input_v))
            elif i == 3:
                # ...
                sub_model_i.load_weights(self.folder_model + name + '_clean_upgrade.h5')
            else:
                sub_model_i.load_weights(self.folder_model + name + '_{}.h5'.format(i))

    def save(self, name=None):
        if name is None:
            name = self.file_name

        for i, sub_model_i in enumerate(self.sub_models):
            if i == 0:
                sub_model_i.save_weights(self.folder_model + name + '_{}_v{}.h5'.format(i, self.input_v))
            elif i == 3:
                sub_model_i.save_weights(self.folder_model + name + '_clean_upgrade.h5')
            else:
                sub_model_i.save_weights(self.folder_model + name + '_{}.h5'.format(i))


def main_train_adop():
    w = img_rows
    ext_zoom = 0
    
    ext = [ext_zoom, ext_zoom, ext_zoom, 0]

    network_adop_normal = NetworkAdop()
    network_adop_normal.load(also_adop = True)

    def get_xy(dict_data, i):
        img_clean = dict_data.get_img_clean()
        img_ir = dict_data.get_img_ir()
        img_rgb = dict_data.get_img_rgb()
        img_y = dict_data.get_img_y()
        mask_annot_test = (np.greater(np.sum(img_y, axis=2), 0)).astype(int)
        data = tools_data.Data(img_clean)
    
        x_list_test = data.img_mask_to_x([img_clean, img_ir, img_rgb, img_y], mask_annot_test, w=w,
                                         ext=ext, name='unet_0_v{}'.format(i), bool_new=False)
    
        x_clean = x_list_test[0]
        x_rgb = x_list_test[2]
        x_ir = x_list_test[1]
        y = x_list_test[3]
        
        y_all_class_0 = np.zeros(np.shape(y))
        
        y_all_class_0[..., 0] = 1
    
        return x_clean, x_rgb, x_ir, y_all_class_0

    # x_clean_tr, x_rgb_tr, x_ir_tr, y_train = get_xy(dict_data)

    # a
    x_clean_tr = None
    x_rgb_tr = None
    x_ir_tr = None
    y_tr = None

    dict_data_list = [main_lamb.MainData(set='zach_small', w=w), main_lamb.MainData(set='hand_small', w=w)]

    for i, dict_data_i in enumerate(dict_data_list):
        if i == 0:
            x_clean_tr, x_rgb_tr, x_ir_tr, y_tr = get_xy(dict_data_i, i)
    
        else:
            x_clean_tr_i, x_rgb_tr_i, x_ir_tr_i, y_tr_i = get_xy(dict_data_i, i)
        
            x_clean_tr = np.concatenate([x_clean_tr, x_clean_tr_i], axis=0)
            x_rgb_tr = np.concatenate([x_rgb_tr, x_rgb_tr_i], axis=0)
            x_ir_tr = np.concatenate([x_ir_tr, x_ir_tr_i], axis=0)
            y_tr = np.concatenate([y_tr, y_tr_i], axis=0)
    # b
    
    # TODO: redundant information a bit removed
    n_train = 1000
    x_clean_tr = x_clean_tr[:n_train, ...]
    x_rgb_tr = x_rgb_tr[:n_train, ...]
    x_ir_tr = x_ir_tr[:n_train, ...]
    y_tr = y_tr[:n_train, ...]

    x_train = [x_clean_tr, x_rgb_tr, x_ir_tr]

    network_adop_normal.train(x_train, y_tr, save=True, epochs=1000)


def main_train():
    if 0:
        ext_zoom = (572-388)//2
        w = 388
    else:
        w = img_rows
        ext_zoom = 0
    
    network = Network()
    ext_zoom = network.n_pad//2
    if 1:
        network.load()
        
    ext = [(ext_zoom, network.n_pad-ext_zoom)]*3 + [0]

    def get_xy(dict_data, i):
        img_clean = dict_data.get_img_clean()
        img_ir = dict_data.get_img_ir()
        img_rgb = dict_data.get_img_rgb()
        img_y = dict_data.get_img_y()
        mask_annot_test = (np.greater(np.sum(img_y, axis=2), 0)).astype(int)
        data = tools_data.Data(img_clean)
    
        x_list_test = data.img_mask_to_x([img_clean, img_ir, img_rgb, img_y], mask_annot_test, w=w,
                                         ext=ext, name='unet_0_v{}'.format(i), bool_new=True, n_max=1000)
    
        x_clean = x_list_test[0]
        x_rgb = x_list_test[2]
        x_ir = x_list_test[1]
        y = x_list_test[3]
    
        return x_clean, x_rgb, x_ir, y

    # x_clean_tr, x_rgb_tr, x_ir_tr, y_train = get_xy(dict_data)

    # a
    x_clean_tr = None
    x_rgb_tr = None
    x_ir_tr = None
    y_tr = None

    dict_data_list = [main_lamb.MainData(set='zach_small', w=w), main_lamb.MainData(set='hand_small', w=w)]

    for i, dict_data_i in enumerate(dict_data_list):
        if i == 0:
            x_clean_tr, x_rgb_tr, x_ir_tr, y_tr = get_xy(dict_data_i, i)

        else:
            x_clean_tr_i, x_rgb_tr_i, x_ir_tr_i, y_tr_i = get_xy(dict_data_i, i)

            x_clean_tr = np.concatenate([x_clean_tr, x_clean_tr_i], axis=0)
            x_rgb_tr = np.concatenate([x_rgb_tr, x_rgb_tr_i], axis=0)
            x_ir_tr = np.concatenate([x_ir_tr, x_ir_tr_i], axis=0)
            y_tr = np.concatenate([y_tr, y_tr_i], axis=0)
    # b

    x_train = [x_clean_tr, x_rgb_tr, x_ir_tr]

    network.train(x_train, y_tr, save=True, epochs=10000)

def main():
    if 0:
        ext_zoom = (572-388)//2
        w = 388
    else:
        w = img_rows
        ext_zoom = 0

    if 0:
        # training
        main_train()

    if 0:
        # train adoptation
        main_train_adop()

    """ prediction """

    dict_data = main_lamb.MainData(set='zach_small', w=w)
    # dict_data = main_lamb.MainData(set='hand_big', w=w)
    # dict_data = main_lamb.MainData(set='hand_small', w=w)
    
    network = Network()
    network.summary()

    ext_zoom = (network.n_pad)//2
    if 1:
        network.load()
        # network.save()
        # network.load2()

    img_clean = dict_data.get_img_clean()
    img_rgb = dict_data.get_img_rgb()
    img_ir = dict_data.get_img_ir()
    img_y = dict_data.get_img_y()

    axes = (0, 1)
    
    # n_rot = 3
    # flip = 1
    # for _ in range(n_rot):
    #     img_clean = np.rot90(img_clean, axes=axes)
    #     img_rgb = np.rot90(img_rgb, axes=axes)
    #     img_ir = np.rot90(img_ir, axes=axes)
    #     img_y = np.rot90(img_y, axes=axes)
    #
    # if flip:
    #     img_clean = np.flip(img_clean, axis=0)
    #     img_rgb = np.flip(img_rgb, axis=0)
    #     img_ir = np.flip(img_ir, axis=0)
    #     img_y = np.flip(img_y, axis=0)

    data = tools_data.Data(img_clean, w=w)
    
    img_in = [img_clean, img_rgb, img_ir]
    img_out = img_y
    
    import link_to_soliton.metrics.roc as roc
    
    def foo(img_in_i , img_out_i):
        ext_tuple = (ext_zoom, network.n_pad-ext_zoom)

        data = tools_data.Data(img_in_i[0], w=w)
        
        x_in_i = [data.img_to_x2(a, ext=ext_tuple) for a in img_in_i]

        y_pred_i = network.predict(x_in_i)

        pred_imgs_i = data.y_to_img2(y_pred_i)
    
        return roc.curve2(y_pred=pred_imgs_i, y_true=img_out_i)
    
    if 0: # calculate all the AUCs
        auc_normal = None
        auc_all = []
        auc_upside_down = []
        
        for i_rot in range(4):
            for i_flip in range(2):
                img_in_i = [np.copy(a) for a in img_in]
                img_out_i = np.copy(img_out)
    
                for _ in range(i_rot):
                    img_in_i = [np.rot90(a, axes=axes) for a in img_in_i]
                    img_out_i = np.rot90(img_out_i, axes=axes)
    
                for _ in range(i_flip):
                    img_in_i = [np.flip(a, axis=0) for a in img_in_i]
                    img_out_i = np.flip(img_out_i, axis=0)
                
                auc_i = foo(img_in_i ,img_out_i)
                
                if i_rot == 0 and i_flip == 0:
                    auc_normal = auc_i
    
                auc_all.append(auc_i)
                
                if (i_flip == 1 and i_rot == 0) or (i_flip == 0 and i_rot == 2):
                    auc_upside_down.append(auc_i)
        
        print('AUC_normal = {}'.format(auc_normal))
        print('AUC_all = {}'.format(np.mean(auc_all)))
        print('AUC_upside_down = {}'.format(np.mean(auc_upside_down)))
    
    ext_tuple = (ext_zoom, network.n_pad-ext_zoom)

    x_clean = data.img_to_x2(img_clean, ext=ext_tuple)
    x_rgb = data.img_to_x2(img_rgb, ext=ext_tuple)
    x_ir = data.img_to_x2(img_ir, ext=ext_tuple)
    
    x_clean_0 = data.img_to_x2(img_clean, ext=0)
    
    x_in = [x_clean, x_rgb, x_ir]
    y_pred = network.predict(x_in)
    
    pred_imgs = data.y_to_img2(y_pred)
   
   
    import link_to_soliton.metrics.roc as roc
    roc.curve(y_pred=pred_imgs, y_true=img_y)
    # TODO check if both are correct!
    roc.curve2(y_pred=pred_imgs, y_true=img_y)
    
    if 1:
        y_true = data.img_to_x2(img_y, ext=0)

        dice = K.eval(keras_ipi.metrics.dice_with_0_labels(y_true=y_true, y_pred=y_pred))
        print('DICE = {}'.format(dice))

        jaccard = K.eval(keras_ipi.metrics.jaccard_with_0_labels(y_true=y_true, y_pred=y_pred))
        print('jaccard = {}'.format(jaccard))
    
    # add difference 2:

    def get_pred_rgb(img_clean, pred_img):
        cyan = [0, 1, 1]
        pred_rgb = np.copy(img_clean)
        pred_rgb[pred_img[:, :, 1] > 0.5, :] = cyan
        return pred_rgb

    segmentation_rgb = get_pred_rgb(img_clean, pred_imgs)

    image_tools.save_im(segmentation_rgb,
                        '/home/lameeus/data/ghent_altar/output/shift_invariant_unet/segm_zach_cyan.tif')
    
    list_rgbs = [segmentation_rgb]
    list_titles = ['result']

    if 0:
        model_clean = network.sub_models[-1]
        
        y_pred_clean = model_clean.predict(x_clean_0)
    
        pred_imgs_clean = data.y_to_img2(y_pred_clean)
    
        cap01 = np.copy(pred_imgs_clean)
        cap01[cap01 > 1] = 1
        cap01[cap01 < 0] = 0
    
        diff = cap01 - img_clean
        diff = (diff * 100) + 0.5
        diff[diff > 1] = 1
        diff[diff < 0] = 0
    
        list_rgbs.extend([cap01, diff])
        list_titles.extend(['clean preprocessing', 'difference'])

    if 0:  # network adoptation
        network_adop = NetworkAdop()
        network_adop.load()
    
        y_pred_adop = network_adop.predict(x_in)
    
        pred_imgs_adop = data.y_to_img2(y_pred_adop)
    
        # TODO keep
        pred_imgs_adop[pred_imgs_adop > 1] = 1
        pred_imgs_adop[pred_imgs_adop < 0] = 0
        # pred_imgs_adop = pred_imgs_adop/np.max(pred_imgs_adop)
    
        diff2 = pred_imgs_adop - img_clean
    
        list_rgbs.append(pred_imgs_adop)
        list_titles.append('adoptation')
    
        if 0:
            list_rgbs.append(diff2)
            list_titles.append('check difference')

    tools_plot.imshow(list_rgbs, title=list_titles)


if __name__ == '__main__':
    main()
