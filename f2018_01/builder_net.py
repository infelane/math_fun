import numpy as np
from keras.layers import Flatten, Input, Dense, Reshape, Concatenate, Conv2D,\
    MaxPooling2D, Conv2DTranspose, Cropping2D
from keras.models import Model


def construct_input_layer(x):
    """
    Build the input layer, concatenate if necessary
    :param x:
    :return:
    """

    if isinstance(x, list):
    
        inputs = []
        for x_i in x:
            input_shape = np.shape(x_i)[1:]
            layer_in_i = Input(shape=input_shape)
            inputs.append(layer_in_i)
        
        layer_in = Concatenate()(inputs)

    else:
        input_shape = np.shape(x)[1:]
    
        inputs = Input(shape=input_shape)
        layer_in = inputs
    
    return inputs, layer_in


def simplest(x, y):
    inputs, layer_in = construct_input_layer(x)
    
    output_shape = np.shape(y)[1:]
    output_flat = np.prod(output_shape)

    layer_1 = Flatten()(layer_in)
    layer_2 = Dense(units=output_flat, activation='softmax')(layer_1)
    outputs = Reshape(output_shape)(layer_2)
    
    model = Model(inputs, outputs)
    
    return model


def shallow_fcc(x, y, units=10):
    inputs, layer_in = construct_input_layer(x)
    
    output_shape = np.shape(y)[1:]
    output_flat = np.prod(output_shape)

    layer_1 = Flatten()(layer_in)
    layer_15 = Dense(units=units, activation='tanh')(layer_1)
    layer_2 = Dense(units=output_flat, activation='softmax')(layer_15)
    outputs = Reshape(output_shape)(layer_2)

    model = Model(inputs, outputs)

    return model


def simplest_cnn(x, y):
    inputs, layer_in = construct_input_layer(x)

    output_shape = np.shape(y)[1:]

    outputs = Conv2D(filters=output_shape[-1], kernel_size=(1, 1), activation='softmax')(layer_in)

    model = Model(inputs, outputs)

    return model


def shallow_cnn(x, y, units=10):
    inputs, layer_in = construct_input_layer(x)

    output_shape = np.shape(y)[1:]

    layer_1 = Conv2D(filters=units, kernel_size=(1, 1), activation='tanh')(layer_in)
    outputs = Conv2D(filters=output_shape[-1], kernel_size=(1, 1), activation='softmax')(layer_1)
    
    model = Model(inputs, outputs)

    return model


def simple_unet(x, y):
    """
    simple unet architecture
    :param x:
    :param y:
    :return:
    """
    inputs, layer_in = construct_input_layer(x)
    output_shape = np.shape(y)[1:]
    

    layer_1 = _gen_conv()(layer_in)
    down1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer_1)
    layer_2_0 = _gen_conv()(down1)
    up1 = Conv2DTranspose(filters=10, kernel_size=(2, 2), strides=(2, 2))(layer_2_0)
    conc1 = Concatenate()([Cropping2D(((2, 2), (2, 2)))(layer_1), up1])
    outputs = Conv2D(filters=output_shape[-1], kernel_size=(1, 1), activation='softmax')(conc1)

    model = Model(inputs, outputs)

    return model

def simple_unet_shift(x, y):
    """
    Shift invariant U-Net,
    Conv2dtranspose are replaced by regular Conv2D, results in need for bigger input size
    :param x:
    :param y:
    :return:
    """
    inputs, layer_in = construct_input_layer(x)
    output_shape = np.shape(y)[1:]
    
    layer_1 = _gen_conv()(layer_in)
    down1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1),
                         # padding='same'
                         )(layer_1)
    layer_2_0 = _gen_conv(dr=2)(down1)
    
    # conv2dtranspose only works for upsampling, at same resolution you always have padding!
    up1 = Conv2D(filters=10, kernel_size=(2, 2), strides=(1, 1),
                 # padding='same'
                 )(layer_2_0)
    
    # TODO note how the cropping is bigger!, this is to solve the padding
    conc1 = Concatenate()([Cropping2D(((3, 3), (3, 3)))(layer_1), up1])
    outputs = Conv2D(filters=output_shape[-1], kernel_size=(1, 1), activation='softmax')(conc1)
    
    model = Model(inputs, outputs)

    return model


def complex_unet(x, y):
    """
        Regular U-Net
        :param x:
        :param y:
        :return:
        """
    
    inputs, layer_in = construct_input_layer(x)
    output_shape = np.shape(y)[1:]

    layer_1 = Cropping2D(((3, 3), (3, 3)))(layer_in)
    layer_1 = _gen_conv(name='left1')(layer_1)
    down1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                         # padding='same'
                         )(layer_1)
    layer_2_0 = _gen_conv(dr=1, name='left2')(down1)
    down2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                         # padding='same'
                         )(layer_2_0)
    layer_3_0 = _gen_conv(dr=1, name='left3')(down2)
    up2 = Conv2DTranspose(filters=10, kernel_size=(2, 2), name='up2', strides=(2, 2)
                          )(layer_3_0)

    crop_val = ((2, 2), (2, 2))
    conc2 = Concatenate()([Cropping2D(crop_val)(layer_2_0), up2])

    layer_right_1 = _gen_conv(dr=1)(conc2)

    up1 = Conv2DTranspose(filters=10, kernel_size=(2, 2), name='up1', strides=(2, 2)
                          )(layer_right_1)

    # TODO note how the cropping is bigger!, this is to solve the padding
    crop_val = ((8, 8), (8, 8))
    conc1 = Concatenate()([Cropping2D(crop_val)(layer_1), up1])
    outputs = Conv2D(filters=output_shape[-1], kernel_size=(1, 1), activation='softmax')(conc1)
    outputs = Cropping2D(((1, 1), (1, 1)))(outputs)

    model = Model(inputs, outputs)
    
    return model


def complex_unet_shift(x, y):
    """
    Shift invariant U-Net,
    Conv2dtranspose are replaced by regular Conv2D, results in need for bigger input size
    :param x:
    :param y:
    :return:
    """
    
    filters = 10
    
    inputs, layer_in = construct_input_layer(x)
    output_shape = np.shape(y)[1:]
    
    layer_1 = Cropping2D(((1, 1), (1, 1)))(layer_in)
    layer_1 = _gen_conv(name='left1')(layer_1)
    down1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1),
                         # padding='same'
                         )(layer_1)
    layer_2_0 = _gen_conv(dr=2, name='left2')(down1)
    # max with 2x2 with dil_rate=2 becomes a 3x3
    down2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                         # padding='same'
                         )(layer_2_0)
    layer_3_0 = _gen_conv(dr=2 ** 2, name='left3')(down2)
    # conv2dtranspose only works for upsampling, at same resolution you always have padding!
    up2 = Conv2D(filters=filters, dilation_rate=(2, 2), kernel_size=(2, 2), strides=(1, 1),
                 # padding='same',
                 name = 'up2'
                 )(layer_3_0)
    crop_val = ((6, 6), (6, 6))
    conc2 = Concatenate()([Cropping2D(crop_val)(layer_2_0), up2])
    
    layer_right_1 = _gen_conv(dr=2, name='right_1')(conc2)
    
    # conv2dtranspose only works for upsampling, at same resolution you always have padding!
    up1 = Conv2D(filters=filters, kernel_size=(2, 2), strides=(1, 1),
                 # padding='same',
                 name='up1'
                 )(layer_right_1)
    
    # TODO note how the cropping is bigger!, this is to solve the padding
    crop_val = ((11, 11), (11, 11))
    conc1 = Concatenate()([Cropping2D(crop_val)(layer_1), up1])
    outputs = Conv2D(filters=output_shape[-1], kernel_size=(1, 1), activation='softmax')(conc1)
    
    model = Model(inputs, outputs)
    
    return model


def _gen_conv(act='elu', name=None, dr=1):
    return Conv2D(filters=10, kernel_size=(3, 3), dilation_rate=(dr, dr), activation=act,
                  # padding= 'same',
                  name=name
                  )
