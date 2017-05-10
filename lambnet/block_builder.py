import keras
from keras import backend as K
import lambnet


def stack(layers_foo):
    model = lambnet.models.Sequential()

    # TODO
    # input_shape = list(layers_foo.layer_size[0]) + [layers_foo.kernel_depth[0]]
    input_shape = [22, 22, 7]
    
    beta1 = 0.001
    l1 = keras.regularizers.l1(beta1)
 
    # CROPPING LAYER
    layer_i = lambnet.layer.Cropping2D(input_shape = input_shape,
                                       cropping=((5, 5), (5, 5)))
    model.add(layer_i)
 
    for index, layer_type in enumerate(layers_foo.layer_types):
        print(layer_type)

        filters = layers_foo.kernel_depth[index + 1]
        kernel_size = layers_foo.kernel_size[index]

        if layer_type == 'conv':
            activation = K.elu # https://arxiv.org/pdf/1511.07289.pdf
            layer_i = keras.layers.Conv2D(filters, kernel_size, padding='valid', activation = activation,
                                          input_shape = input_shape,
                                          kernel_initializer='glorot_normal', kernel_regularizer=l1,
                                          
                                          ) # same as Convolution2D
                                    
        elif layer_type == 'softmaxsconv':
            layer_i = keras.layers.Conv2D(filters, kernel_size, padding='valid', activation= 'softmax',
                                          input_shape=input_shape,
                                          kernel_initializer='glorot_normal'#, kernel_regularizer=l1
                                          )
            
        else:
            errmsg = "Unknown layer type: " + layer_type
            raise(ValueError(errmsg))

        # todo
        # model.add(keras.layers.Dropout(0.9, input_shape = input_shape))
        model.add(layer_i)
        
    return model
   