import numpy as np
from keras.layers import Flatten, Dense, Reshape, Input, Concatenate, Conv2D, Cropping2D
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from link_to_keras_ipi.metrics import jaccard_with_0_labels

from f2018_01 import data_sets


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


def simplest2():
    w = 36

    input_shape = [w, w ,3]
    
    inputs = Input(shape=input_shape)
    layer_in = inputs
    
    layer_1 = Cropping2D([[13, 13],[13, 13]])(layer_in)
    layer_2 = Conv2D(100, (1, 1), activation='sigmoid')(layer_1)
    outputs = Conv2D(2, (1, 1), activation='softmax')(layer_2)
    
    model = Model(inputs, outputs)
    
    return model

def trained_net(bool_train = False):
    epochs = 100
    ext_tot = 0

    path = '/home/lameeus/data/NO_PLACE/'
    
    if 0:
        (x_train, y_train), (x_test, y_test), (x_val, y_val) = data_sets.load_art(ext_tot=ext_tot)

        x_train = x_train[0]
        x_test = x_test[0]

        np.save(path + 'x_train', x_train)
        np.save(path + 'x_test', x_train)
        np.save(path + 'y_train', y_train)
        np.save(path + 'y_test', y_train)

    else:
        x_train = np.load(path + 'x_train.npy')
        x_test = np.load(path + 'x_test.npy')
        y_train = np.load(path + 'y_train.npy')
        y_test = np.load(path + 'y_test.npy')

    model = simplest2()

    # compile the model
    model.compile(optimizer=Adam(),
                  # loss='mse',
                  loss=categorical_crossentropy,
                  metrics=[
                      jaccard_with_0_labels])  # accuracy is super bad metric with some outputs being [0, 0] (no annotation)
    path_weight = '/home/lameeus/data/NO_PLACE/weights.h5'

    model.load_weights(path_weight)

    if bool_train:
        # Train again

        model.fit(x_train, y_train,
                  epochs=epochs,
                  validation_data=[x_test, y_test],
                  # callbacks=[checkpoint, tb, drd]
                  )
        
        model.save_weights(path_weight)

    return model
