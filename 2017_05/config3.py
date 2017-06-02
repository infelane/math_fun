import keras_ipi
import numpy as np


class flag():
    bool_prev = True   # Load previous
    lr =1.0e-4  # learning rate
    beta =0.9    # standard 0.9
    epochs = 0
    batch_size = 100
    

def nn():
    layer_width = 8
    
    depth = 6
    kernel_width = 3
    kernel_dep = 2
    
    layer_size_in = [layer_width + 2*7, layer_width + 2*7]
    layer_size_out = [layer_width, layer_width]

    kernel_size = [[kernel_width, kernel_width]] * (depth - 1) + [[]]*5 + [[kernel_width, kernel_width]]
    kernel_depth = [7] + [kernel_dep]*(depth-1) + [0] + [1] + [2] + [3] + [4] + [2]
    
    # layer_types = ['convsame'] * (depth - 1) + ['softmaxsame']
    layer_types = ['conv'] * (depth - 1) + ['concat']*5 + ['conv']
    # layer_types = ['conv'] * (depth - 1) + ['conv']
    
    cost = 'wxentropy'  # 'wxentropy'
    
    class_rate = 20.0  # Fixed, should be calculated from the input data
    k = 2.0

    # return keras_ipi.layers.LayerConfig(None, kernel_size, kernel_depth, layer_types,
    #                                  layer_size_in, layer_size_out,
    #                                  cost=cost, k=k, r=class_rate)


    layers = []
    for i_layer in range(len(layer_types)):
        # if layer_types[i_layer] == 'conv':
        layer_config_i = {'type': layer_types[i_layer]
                          }
        
        if layer_config_i['type'] == 'conv':
            layer_config_i.update({'depth': kernel_depth[i_layer + 1],
                                   'kernel': kernel_size[i_layer]
                                   })
        
        if layer_config_i['type'] == 'concat':
            # TODO select layer to concat with
            layer_config_i.update({'value': kernel_depth[i_layer + 1]
                                   })

            
        #last layer
        if i_layer == (len(layer_types) - 1):
            layer_config_i.update({'activation': 'softmax'})
            
        layers.append(layer_config_i)

    layer_config = {}
    layer_config.update({'input_size': layer_size_in})
    layer_config.update({'input_depth': kernel_depth[0]})
    layer_config.update({'layers': layers})
    layer_config.update({'depth' : depth})
    
    layer_config.update({'k': k})
    layer_config.update({'r': class_rate})
    
    return layer_config
