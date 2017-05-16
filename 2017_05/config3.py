import keras_ipi


class flag():
    bool_prev = True   # Load previous
    lr = 1.0e-1     # learning rate
    epochs = 0
    

def nn():
    layer_width = 8
    
    depth = 1
    kernel_width = 3
    kernel_dep = 20
    
    layer_size_in = [layer_width + 2*7, layer_width + 2*7]
    layer_size_out = [layer_width, layer_width]

    kernel_size = [[kernel_width, kernel_width]] * depth
    kernel_depth = [7] + [kernel_dep]*(depth-1) + [2]
    
    # layer_types = ['convsame'] * (depth - 1) + ['softmaxsame']
    layer_types = ['conv'] * (depth - 1) + ['softmaxsconv']
    
    cost = 'wxentropy'  # 'wxentropy'
    k = 6.0
    class_rate = 30.0  # Fixed, should be calculated from the input data

    return keras_ipi.layer.LayerConfig(None, kernel_size, kernel_depth, layer_types,
                                     layer_size_in, layer_size_out,
                                     cost=cost, k=k, r=class_rate)
