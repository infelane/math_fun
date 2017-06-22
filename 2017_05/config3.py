import keras_ipi
import numpy as np


class flag():
    bool_prev = True   # Load previous
    lr =0.3e-4 # learning rate
    beta =0.90    # standard 0.9
    epochs = 0
    batch_size = 100
    

def nn():
    layer_width = 8
    
    depth = 6
    kernel_width = 3
    kernel_dep = 10
    
    output_depth = 2
    
    layer_size_in = [layer_width + 2*7, layer_width + 2*7]
    layer_size_out = [layer_width, layer_width]

    kernel_size = [[kernel_width, kernel_width]] * (depth - 1) + [[kernel_width, kernel_width]] + [[]]*6 + [[1, 1]] + [[]]
    kernel_depth = [7] + [kernel_dep]*(depth-1) + [kernel_dep] +  [0] + [1] + [2] + [3] + [4] + [5] + [output_depth] + [2]
    
    # layer_types = ['convsame'] * (depth - 1) + ['softmaxsame']
    layer_types = ['conv'] * (depth - 1) + ['conv'] + ['concat']*6 + ['conv'] + ['max_half']
    # layer_types = ['conv'] * (depth - 1) + ['conv']
    
    cost = 'wxentropy'  # 'wxentropy'
    
    # Can be calculated from print(np.mean(Y_train, axis = (0, 1, 2)))
    class_rate = 21.5  # Fixed, should be calculated from the input data
    k = 4.0

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
        
        elif layer_config_i['type'] == 'concat':
            # TODO select layer to concat with
            layer_config_i.update({'value': kernel_depth[i_layer + 1]
                                   })
            
        elif layer_config_i['type'] == 'max_half':
            layer_config_i.update({'depth': kernel_depth[i_layer + 1 - 1]})

            
        #last layer
        if i_layer == (len(layer_types) - 2):
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


def nn2():
    layer_width = 8
    input_depth = 7
    output_depth = 2
    kernel_dep = 10

    # Can be calculated from print(np.mean(Y_train, axis = (0, 1, 2)))
    class_rate = 21.5  # Fixed, should be calculated from the input data
    k = 4.0
    
    layer_size_in = [layer_width + 2 * 7, layer_width + 2 * 7]
    
    layers = []
    
    layer_config_i = {'type': 'conv'}
    layer_config_i.update({'depth': kernel_dep,
                           'kernel': 3,
                           'stride': 1
                           })
    layers.append(layer_config_i)
    
    layer_config_i = {'type': 'conv'}
    layer_config_i.update({'depth': kernel_dep,
                           'kernel': 3,
                           'stride': 2
                           })
    layers.append(layer_config_i)
    
    layer_config_i = {'type': 'conv'}
    layer_config_i.update({'depth': kernel_dep,
                           'kernel': 3,
                           'stride': 4,
                           })
    layers.append(layer_config_i)


    layer_config_i = {'type': 'concat'}
    layer_config_i.update({'value': 0
                           })
    layers.append(layer_config_i)
    
    layer_config_i = {'type': 'concat'}
    layer_config_i.update({'value': 1
                           })
    layers.append(layer_config_i)
    
    layer_config_i = {'type': 'concat'}
    layer_config_i.update({'value': 2
                           })
    layers.append(layer_config_i)
    
    layer_config_i = {'type': 'conv'}
    layer_config_i.update({'depth': output_depth,
                           'kernel': 1,
                           'stride': 1,
                           'activation' : 'softmax'
                           })
    layers.append(layer_config_i)
    
    # kernel_width = 3
    # kernel_dep = 10
    #

    #
    #
    # kernel_size = [[kernel_width, kernel_width]] * (depth - 1) + [[kernel_width, kernel_width]] + [[]] * 6 + [
    #     [1, 1]] + [[]]
    # kernel_depth = [7] + [kernel_dep] * (depth - 1) + [kernel_dep] + [0] + [1] + [2] + [3] + [4] + [5] + [
    #     output_depth] + [2]
    #
    # # layer_types = ['convsame'] * (depth - 1) + ['softmaxsame']
    # layer_types = ['conv'] * (depth - 1) + ['conv'] + ['concat'] * 6 + ['conv'] + ['max_half']
    # # layer_types = ['conv'] * (depth - 1) + ['conv']
    #
    # cost = 'wxentropy'  # 'wxentropy'

    layer_config = {}
    layer_config.update({'input_size': layer_size_in})
    layer_config.update({'input_depth': input_depth})
    layer_config.update({'layers': layers})
    # layer_config.update({'depth': depth})
    layer_config.update({'k': k})
    layer_config.update({'r': class_rate})

    return layer_config
