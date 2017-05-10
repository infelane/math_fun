import lambnet

class flag():
    bool_prev = True   # Load previous
    lr = 1.0e-4         # learning rate
    

def nn():
    layer_width = 8
    
    depth = 2
    kernel_width = 3
    kernel_dep = 2
    
    layer_size_in = [layer_width + 2*7, layer_width + 2*7]
    layer_size_out = [layer_width, layer_width]

    kernel_size = [[kernel_width, kernel_width]] * depth
    kernel_depth = [7] + [kernel_dep] + [2]
    
    # layer_types = ['convsame'] * (depth - 1) + ['softmaxsame']
    layer_types = ['conv'] * (depth - 1) + ['softmaxsconv']
    
    cost = 'wxentropy'  # 'wxentropy'
    k = 5
    class_rate = 20  # Fixed, should be calculated from the input data

    return lambnet.layer.LayerConfig(None, kernel_size, kernel_depth, layer_types,
                                     layer_size_in, layer_size_out,
                                     cost=cost, k=k, r=class_rate)
