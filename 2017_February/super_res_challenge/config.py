# All config

import lambnet.layer.LayerConfig as LayerConfig

# Settings
class FLAGS1():
    lr = 1.0e-6
    batch_size = 100
    load_prev = True
    train = True
    training_epochs = 1
    ipi_dir = '/ipi/private/lameeus/data/super_res/net2/'
    checkpoint_dir = ipi_dir + 'wb/'
    main_dir = '/scratch/lameeus/NTIRE17/lameeus/'
    # patches_dir = main_dir + 'patches_64_x2/'
    summary_dir = main_dir + 'summary/'
    checkpoint_steps_size = 1
    depth_train = None    # set to None if not interested
    clip_val = 1.0e10#1.0e-5


def layer_config1():
    layer_size = [[3, 3], [1, 1], [2, 2]]
    kernel_size = [[3, 3], [2, 2]]
    kernel_depth = [3, 100, 3]
    layer_types = ['convrgb', 'deconv']

    return LayerConfig(layer_size, kernel_size, kernel_depth, layer_types)


def layer_config2():
    layer_size = [[5, 5], [1, 1], [1, 1]]
    kernel_size = [[5, 5], [1, 1]]
    kernel_depth = [1, 100, 1]
    layer_types = ['conv', 'conv']

    return LayerConfig(layer_size, kernel_size, kernel_depth, layer_types)


def layer_config3():
    depth = 10
    depth1 = 7
    depth2 = depth - depth1
    layer_width = 32
    kernel_width = 3
    kernel_dep = 50
    
    layer_size_1 = [[layer_width + 2*ext_i, layer_width + 2*ext_i] for ext_i in range(depth1, 0, -1)]
    layer_size = layer_size_1 + [[layer_width, layer_width]] * (depth2+1)
    kernel_size = [[kernel_width, kernel_width]] * depth1 + [[1, 1]] * depth2

    # kernel_size = [[4, 4]] * depth
    # kernel_size = [[3, 3], [3, 3], [3, 3], [3, 3]]
    # kernel_depth = [1] + [kernel_dep] * (depth -1) + [1]
    kernel_depth = [1] + [kernel_dep] * (depth1) + [kernel_dep*2] * (depth2 - 1) + [1]

    # layer_types = ['convsame'] * (depth-1) + ['last']
    layer_types = ['conv']*(depth-1) + ['last']
       
    print(layer_size)
    print(kernel_size)
    print(kernel_depth)
    print(layer_types)
    
    return LayerConfig(layer_size, kernel_size, kernel_depth, layer_types)


def layer_config4():
    new_width = 90
    a = layer_config3()
    a.layer_size = [[new_width, new_width] for layer in a.layer_size]
    return a


def layer_config_fus0():
    patch_width = 32
    layer_size = [[patch_width+ 4, patch_width+ 4], [patch_width+ 2, patch_width+ 2], [patch_width, patch_width], [patch_width, patch_width]]
    kernel_size = [[3, 3], [3, 3], [1, 1]]
    kernel_depth = [12] + [50] + [50] + [1]
    layer_types = ['conv'] + ['conv'] + ['fus_conv']
    return LayerConfig(layer_size, kernel_size, kernel_depth, layer_types)



