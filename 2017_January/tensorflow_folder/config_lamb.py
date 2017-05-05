from tf_lamb import tf_network
import os
import sys

folder_loc = '/ipi/private/lameeus/private_Documents/python/2017_February/super_res_challenge'
cmd_subfolder = os.path.realpath(folder_loc)
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import config

# Settings new
class FLAGS1(object):
    train = False
    training_epochs = 10000 # amount of times it goes through the whole training data set
    checkpoint_steps_size = 1
    main_dir = '/scratch/lameeus/data/lamb/'
    ipi_dir = '/ipi/private/lameeus/data/lamb/'
    checkpoint_dir = ipi_dir + 'wb_drop3/' #'wb/' #
    summary_dir = main_dir + 'summary_drop3/'
    load_prev = True
    lr = 2.0e-7
    mom = 0.9  # momentum higher with dropout!
    batch_size = 1000
    dropout = 1.0
    # mom = 0.99  # momentum higher with dropout!
    # batch_size = 1000
    # dropout = 0.5
    

# # Settings old
# class FLAGS1(object):
#     train = True
#     training_epochs = 10 # amount of times it goes through the whole training data set
#     checkpoint_steps_size = 1
#     main_dir = '/scratch/lameeus/data/lamb/'
#     ipi_dir = '/ipi/private/lameeus/data/lamb/'
#     checkpoint_dir = ipi_dir + 'wb_lab3/' #'wb/' #
#     summary_dir = main_dir + 'summary/'
#     load_prev = True
#     lr = 1.0e-6
#     batch_size = 1000
#     dropout = 1.0
#
    
# My standard NN
def NN1():
    layer_size = [[21, 21], [17, 17], [1, 1]]
    kernel_size = [[5, 5], [17, 17]]
    kernel_depth = [7, 5, 2]
    layers = tf_network.Layers(layer_size, kernel_size, kernel_depth)
    return layers


# simpeler NN
def NN2():
    layer_size = [[21, 21], [1, 1]]
    kernel_size = [[21, 21]]
    kernel_depth = [7, 2]
    layers = tf_network.Layers(layer_size, kernel_size, kernel_depth)
    return layers


def NN3():
    layer_size = [[3, 3], [3, 3], [1, 1]]
    kernel_size = [[1, 1], [3, 3]]
    kernel_depth = [7, 10, 2]
    layers = tf_network.Layers(layer_size, kernel_size, kernel_depth)
    return layers

def nn3():
    # depth = 10
    #
    # layer_width = 32
    # kernel_width = 3
    # # layer_size = [[3, 3], [3, 3], [3, 3], [3, 3], [1, 1]]
    # layer_size = [[layer_width, layer_width]]*(depth+1)
    # kernel_size = [[kernel_width, kernel_width]]*(depth)
    # kernel_depth = [7] + [32]*(depth-1)+ [2]
    # layer_types = ['convsame'] + ['convsameflat']*(depth-2) + ['softmaxsame']
    # cost = 'wxentropy' #'wxentropy'
    # k = 1
    # class_rate = 30 #Fixed, could be calculated from the input data
    
    # depth = 21
    # depth_half = 10
    # layer_width = 32
    # kernel_width = 3
    #
    # layer_size = [[layer_width, layer_width]]*(depth+1)
    # kernel_size = [[1, 1], [kernel_width, kernel_width]]*(depth_half) + [[kernel_width, kernel_width]]
    # kernel_depth = [7] + [32]*(depth-1)+ [2]
    #
    # layer_types = ['convsame', 'convsameflat']*depth_half + ['softmaxsame']
    # cost = 'wxentropy' #'wxentropy'
    # k = 5
    # class_rate = 30 #Fixed, should be calculated from the input data
    
    depth = 10
    layer_width = 200
    kernel_width = 3
    
    layer_size = [[layer_width, layer_width]]*(depth+1)
    kernel_size = [[kernel_width, kernel_width]]*depth
    kernel_depth = [7] + [64]*(depth-1)+ [2]
    
    layer_types = ['conv1']*(depth-1) + ['softmaxsconv']
    
    cost = 'wxentropy' #'wxentropy'
    k = 1
    class_rate = 30 #Fixed, should be calculated from the input data

    return config.LayerConfig(layer_size, kernel_size, kernel_depth, layer_types,
                              cost = cost, k = k, r = class_rate)

def nn4():
    layer_width = 8
    
    depth = 10
    depth1 = 7
    depth2 = depth - depth1
    kernel_width = 3
    kernel_dep = 50
        
    layer_size_1 = [[layer_width + 2*ext_i, layer_width + 2*ext_i] for ext_i in range(depth1, 0, -1)]
    layer_size = layer_size_1 + [[layer_width, layer_width]] * (depth2+1)
    kernel_size = [[kernel_width, kernel_width]] * depth1 + [[1, 1]] * depth2
    kernel_depth = [7] + [kernel_dep] * (depth1) + [100] * (depth2 - 1) + [2]

    # layer_types = ['convsame'] * (depth - 1) + ['softmaxsame']
    layer_types = ['conv']*(depth-1) + ['softmaxsconv']
    
    cost = 'wxentropy'  # 'wxentropy'
    k = 5
    class_rate = 30  # Fixed, should be calculated from the input data
    
    return config.LayerConfig(layer_size, kernel_size, kernel_depth, layer_types,
                              cost=cost, k=k, r=class_rate)

