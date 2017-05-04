import keras

def foo(layers):
    model = keras.models.Sequential()

    for index, layer_type in enumerate(layers.layer_types):
        print(layer_type)
        # model.add()
    
    return model