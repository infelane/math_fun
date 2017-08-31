class flag():
    bool_prev = True   # Load previous
    lr =1.0e-4# learning rate
    beta =0.90    # standard 0.9
    epochs = 100000
    batch_size = 100
    

def nn(width = 8, ext = 26):
    layer_width = width
    input_depth = 7
    output_depth = 2
    kernel_dep = 10
    
    # Can be calculated from print(np.mean(Y_train, axis = (0, 1, 2)))
    class_rate = 21.5  # Fixed, should be calculated from the input data
    k = 4.0
    
    layer_size_in = [layer_width + 2 * ext] * 2
    
    layers = []

    layer_config_i = {'type': 'crop_depth',
                      'depth': (0,3)}   # only the clean RGB
    layers.append(layer_config_i)
    
    layer_config_i = {'type': 'conv'}
    layer_config_i.update({'depth': 10,
                           'kernel': 1,
                           'stride': 1
                           })
    layers.append(layer_config_i)
    
    layer_config_i = {'type': 'conv'}
    layer_config_i.update({'depth': 10,
                           'kernel': 3,
                           'stride': 1
                           })
    layers.append(layer_config_i)
    
    layer_config_i = {'type': 'conv'}
    layer_config_i.update({'depth': 10,
                           'kernel': 5,
                           'stride': 2
                           })
    layers.append(layer_config_i)

    layer_config_i = {'type': 'conv'}
    layer_config_i.update({'depth': 20,
                           'kernel': 1,
                           'stride': 1
                           })
    layers.append(layer_config_i)
    
    # layer_config_i = {'type': 'conv'}
    # layer_config_i.update({'depth': kernel_dep,
    #                        'kernel': 5,
    #                        'stride': 9,
    #                        })
    # layers.append(layer_config_i)
    
    # layer_config_i = {'type': 'concat'}
    # layer_config_i.update({'value': 0
    #                        })
    # layers.append(layer_config_i)
    #
    # layer_config_i = {'type': 'concat'}
    # layer_config_i.update({'value': 1
    #                        })
    # layers.append(layer_config_i)
    #
    # layer_config_i = {'type': 'concat'}
    # layer_config_i.update({'value': 2
    #                        })
    # layers.append(layer_config_i)
    
    layer_config_i = {'type': 'conv'}
    layer_config_i.update({'depth': output_depth,
                           'kernel': 1,
                           'stride': 1,
                           'activation': 'softmax',
                           # 'activation': 'linear'   # TODO
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
    layer_config.update({'width' : width})
    layer_config.update({'ext' : ext})
    # layer_config.update({'input_size': layer_size_in})
    layer_config.update({'input_depth': input_depth})
    layer_config.update({'layers': layers})
    # layer_config.update({'depth': depth})
    layer_config.update({'k': k})
    layer_config.update({'r': class_rate})
    
    return layer_config


def nn2(width=8, ext=26):
    layer_width = width
    input_depth = 7
    output_depth = 2
#     kernel_dep = 10
#
#     # Can be calculated from print(np.mean(Y_train, axis = (0, 1, 2)))
#     class_rate = 21.5  # Fixed, should be calculated from the input data
#     k = 4.0
#
#     layer_size_in = [layer_width + 2 * ext] * 2
#
    layers = []
    
    def conv(depth = 10):
        # to reduce the depth and equalize outputs
        layer_config_i = {'type': 'conv',
                          'kernel': 3,
                          'depth': depth,
                          'activation': 'sigmoid'
                          }
        return layer_config_i
#
    # layer_config_i = {'type': 'conv',
    #                   'kernel' : 3,
    #                   'depth':  10
    #                   }
    
    layer_config_i = conv()
    layers.append(layer_config_i)
    layer_config_i = conv()
    layers.append(layer_config_i)
    layer_config_i = conv()
    layers.append(layer_config_i)
    layer_config_i = conv()
    layers.append(layer_config_i)
    layer_config_i = conv()
    layers.append(layer_config_i)
    layer_config_i = conv()
    layers.append(layer_config_i)
    
    # layer_config_i = {'type': 'conv',
    #                   'kernel': 3,
    #                   'depth':  10
    #                   }
    # layers.append(layer_config_i)
    
    # layer_config_i = {'type': 'conv',
    #                   'kernel': 3,
    #                   'depth':  10
    #                   }
    # layers.append(layer_config_i)
    
    # layer_config_i = {'type': 'conv',
    #                   'kernel': 3,
    #                   'depth':  10
    #                   }
    # layers.append(layer_config_i)
    
    # layer_config_i = {'type': 'conv',
    #                   'kernel': 3,
    #                   'depth':  10
    #                   }
    # layers.append(layer_config_i)
    
    # layer_config_i = {'type': 'conv',
    #                   'kernel': 3,
    #                   'depth':  10
    #                   }
    # layers.append(layer_config_i)
    
    layer_config_i = {'type' : 'concat',
                      'value' : 0}
    layers.append(layer_config_i)

    layer_config_i = {'type': 'concat',
                      'value' : 1}
    layers.append(layer_config_i)

    layer_config_i = {'type': 'concat',
                      'value' : 2}
    layers.append(layer_config_i)
    
    layer_config_i = {'type': 'concat',
                      'value' : 3}
    layers.append(layer_config_i)

    layer_config_i = {'type': 'concat',
                      'value' : 4}
    layers.append(layer_config_i)
    
    layer_config_i = {'type': 'concat',
                      'value': 5}
    layers.append(layer_config_i)
    
    layer_config_i = {'type': 'conv',
                      'kernel': 1,
                      'depth': 2,
                      'activation' : 'softmax'}
    layers.append(layer_config_i)
    
    # layer_config_i = {'type': 'max_half',
    #                   'depth': 2}
    # layers.append(layer_config_i)
    
    
    
    
#                       'depth': (0, 3)}  # only the clean RGB
#     layers.append(layer_config_i)
#
#     layer_config_i = {'type': 'conv'}
#     layer_config_i.update({'depth': 10,
#                            'kernel': 1,
#                            'stride': 1
#                            })
#     layers.append(layer_config_i)
#
#     layer_config_i = {'type': 'conv'}
#     layer_config_i.update({'depth': 10,
#                            'kernel': 3,
#                            'stride': 1
#                            })
#     layers.append(layer_config_i)
#
#     layer_config_i = {'type': 'conv'}
#     layer_config_i.update({'depth': 10,
#                            'kernel': 5,
#                            'stride': 2
#                            })
#     layers.append(layer_config_i)
#
#     layer_config_i = {'type': 'conv'}
#     layer_config_i.update({'depth': 20,
#                            'kernel': 1,
#                            'stride': 1
#                            })
#     layers.append(layer_config_i)
#
#     # layer_config_i = {'type': 'conv'}
#     # layer_config_i.update({'depth': kernel_dep,
#     #                        'kernel': 5,
#     #                        'stride': 9,
#     #                        })
#     # layers.append(layer_config_i)
#
#     # layer_config_i = {'type': 'concat'}
#     # layer_config_i.update({'value': 0
#     #                        })
#     # layers.append(layer_config_i)
#     #
#     # layer_config_i = {'type': 'concat'}
#     # layer_config_i.update({'value': 1
#     #                        })
#     # layers.append(layer_config_i)
#     #
#     # layer_config_i = {'type': 'concat'}
#     # layer_config_i.update({'value': 2
#     #                        })
#     # layers.append(layer_config_i)
#
#     layer_config_i = {'type': 'conv'}
#     layer_config_i.update({'depth': output_depth,
#                            'kernel': 1,
#                            'stride': 1,
#                            'activation': 'softmax',
#                            # 'activation': 'linear'   # TODO
#                            })
#     layers.append(layer_config_i)
#
#     # kernel_width = 3
#     # kernel_dep = 10
#     #
#
#     #
#     #
#     # kernel_size = [[kernel_width, kernel_width]] * (depth - 1) + [[kernel_width, kernel_width]] + [[]] * 6 + [
#     #     [1, 1]] + [[]]
#     # kernel_depth = [7] + [kernel_dep] * (depth - 1) + [kernel_dep] + [0] + [1] + [2] + [3] + [4] + [5] + [
#     #     output_depth] + [2]
#     #
#     # # layer_types = ['convsame'] * (depth - 1) + ['softmaxsame']
#     # layer_types = ['conv'] * (depth - 1) + ['conv'] + ['concat'] * 6 + ['conv'] + ['max_half']
#     # # layer_types = ['conv'] * (depth - 1) + ['conv']
#     #
#     # cost = 'wxentropy'  # 'wxentropy'
#
    layer_config = {}
    layer_config.update({'width': width})
    layer_config.update({'ext': ext})
#     # layer_config.update({'input_size': layer_size_in})
    layer_config.update({'input_depth': input_depth})
    layer_config.update({'layers': layers})
#     # layer_config.update({'depth': depth})
#     layer_config.update({'k': k})
#     layer_config.update({'r': class_rate})
#
    return layer_config


def conv1(depth = 0):
    # to reduce the depth and equalize outputs
    layer_config_i = {'type': 'conv',
                      'kernel': 1,
                      'depth': depth,
                      'activation': 'sigmoid'
                      }
    return layer_config_i
import keras.backend as K


def nn3(width=8, ext=26):
    layer_width = width
    input_depth = 7
    output_depth = 2
    
    layer_depth3 = 10
    layer_depth1 = 3

    layers = []
    
    def conv3():
        layer_config_i = {'type': 'conv',
                          'kernel': 3,
                          'depth': layer_depth3,
                          'activation': K.elu
                          }
        return layer_config_i
    
    layer_config_i = {'type': 'crop_depth',
                      'depth': (3,7)}   # only the clean RGB
    layers.append(layer_config_i)
    
    layers.append(conv1(10))
    layers.append(conv3())
    layers.append(conv1(layer_depth1))
    layers.append(conv3())
    layers.append(conv1(layer_depth1))
    layers.append(conv3())
    layers.append(conv1(layer_depth1))
    layers.append(conv3())
    layers.append(conv1(layer_depth1))
    layers.append(conv3())
    layers.append(conv1(layer_depth1))
    layers.append(conv3())
    layers.append(conv1(layer_depth1))
    layer_config_i = {'type': 'concat',
                      'value': 1}
    
    layers.append(layer_config_i)
    layer_config_i = {'type': 'concat',
                      'value': 3}
    layers.append(layer_config_i)
    layer_config_i = {'type': 'concat',
                      'value': 5}
    layers.append(layer_config_i)
    layer_config_i = {'type': 'concat',
                      'value': 7}
    layers.append(layer_config_i)
    layer_config_i = {'type': 'concat',
                      'value': 9}
    layers.append(layer_config_i)
    layer_config_i = {'type': 'concat',
                      'value': 11}
    layers.append(layer_config_i)
        
    layer_config_i = {'type': 'conv',
                      'kernel': 1,
                      'depth': 2,
                      'activation': 'softmax'}
    layers.append(layer_config_i)
    
    # layer_config_i = {'type': 'max_half',
    #                   'depth': output_depth}
    # layers.append(layer_config_i)
    
    layer_config = {}
    layer_config.update({'width': width})
    layer_config.update({'ext': ext})
    layer_config.update({'input_depth': input_depth})
    layer_config.update({'layers': layers})

    return layer_config


# def nn4(width=8, ext=26):
#     layer_width = width
#     input_depth = 7
#     output_depth = 2
#
#     layer_depth3 = 10
#     layer_depth1 = 3
#
#     layers = []
#
#     def conv1(depth=layer_depth1):
#         # to reduce the depth and equalize outputs
#         layer_config_i = {'type': 'conv',
#                           'kernel': 1,
#                           'depth': depth,
#                           'activation': K.elu# 'sigmoid'   # TODO
#                           }
#         return layer_config_i
#
#     import keras.backend as K
#     def conv3():
#         layer_config_i = {'type': 'conv',
#                           'kernel': 3,
#                           'depth': layer_depth3,
#                           'activation': K.elu
#                           }
#         return layer_config_i
#
#     # layer_config_i = {'type': 'crop_depth',
#     #                   'depth': (0,3)}   # only the clean RGB
#     # layers.append(layer_config_i)
#
#     layers.append(conv3())
#     layers.append(conv3())
#     layers.append(conv3())
#     layers.append(conv3())
#     layers.append(conv3())
#     layers.append(conv3())
#     layer_config_i = {'type': 'concat',
#                       'value': 1}
#
#     layers.append(layer_config_i)
#     layer_config_i = {'type': 'concat',
#                       'value': 2}
#     layers.append(layer_config_i)
#     layer_config_i = {'type': 'concat',
#                       'value': 3}
#     layers.append(layer_config_i)
#     layer_config_i = {'type': 'concat',
#                       'value': 4}
#     layers.append(layer_config_i)
#     layer_config_i = {'type': 'concat',
#                       'value': 5}
#     layers.append(layer_config_i)
#     layer_config_i = {'type': 'concat',
#                       'value': 6}
#     layers.append(layer_config_i)
#
#     layer_config_i = {'type': 'conv',
#                       'kernel': 1,
#                       'depth': 2,
#                       'activation': 'softmax'}
#     layers.append(layer_config_i)
#
#     layer_config = {}
#     layer_config.update({'width': width})
#     layer_config.update({'ext': ext})
#     layer_config.update({'input_depth': input_depth})
#     layer_config.update({'layers': layers})
#
#     return layer_config
