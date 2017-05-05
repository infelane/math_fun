import keras

def stack(layers_foo):
    model = keras.models.Sequential()

    # todo
    input_shape = list(layers_foo.layer_size[0]) + [layers_foo.kernel_depth[0]]
 
    for index, layer_type in enumerate(layers_foo.layer_types):
        print(layer_type)

        filters = layers_foo.kernel_depth[index + 1]
        kernel_size = layers_foo.kernel_size[index]

        if layer_type == 'conv':
     
            layer_i = keras.layers.Conv2D(filters, kernel_size, padding='valid', activation = 'relu',input_shape = input_shape) # same as Convolution2D
                
        # elif if layer_type == 'conv':
        
        elif layer_type == 'softmaxsconv':
            layer_i = keras.layers.Conv2D(filters, kernel_size, padding='valid', activation= 'softmax')
            
        else:
            errmsg = "Unknown layer type: " + layer_type
            raise(ValueError(errmsg))

        model.add(layer_i)
            
    return model