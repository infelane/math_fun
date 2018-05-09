import tensorflow as tf
import tflearn
import numpy as np
import warnings
import os
import sys

# own libraries
from f2017_04 import lam_warnings
# import network_eval

# Abstract class
from abc import ABCMeta


class Layer(metaclass=ABCMeta):
    @staticmethod
    # @abstractmethod
    def my_abstract_staticmethod(self, x):
        ...
    
    def set_input(self):
        ...
    
    def get_output(self):
        ...
    
    def new_func(self):
        ...


from tensorflow.python.ops import nn_ops


class SoftmaxLayerSame(Layer):
    def __init__(self, a_in, kernel_size_i, kernel_depth_i, layer_size_out, index):
        shape_W = (kernel_size_i[0], kernel_size_i[1], kernel_depth_i[0], kernel_depth_i[1])
        shape_b = [1, kernel_depth_i[1]]
                
        self.W = weight_variable(shape_W, index)
        self.b = bias_var(shape_b, index)
        self.vars = {self.W.name: self.W, self.b.name: self.b}
        
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope('layer' + str(index)):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weight'):
                variable_summaries(self.W)
        
        self.a_in = a_in
        
        self.kernel_depth_i = kernel_depth_i
        self.layer_size_out = layer_size_out
        
        out1 = tf.nn.conv2d(self.a_in, self.W, strides=[1, 1, 1, 1], padding='SAME') + self.b
        
        self.out = tf.nn.softmax(out1)
        self.log_out = tf.nn.log_softmax(out1)  # Log and softmax!
        
        tf.nn
    
    def get_log_out(self):
        return self.log_out
    
    def get_output(self):
        return self.out


class SoftmaxLayer(Layer):
    def __init__(self, a_in, kernel_size_i, kernel_depth_i, layer_size_out, index):
        shape_W = (kernel_size_i[0], kernel_size_i[1], kernel_depth_i[0], kernel_depth_i[1])
        shape_b = [1, kernel_depth_i[1]]
        
        self.W = weight_variable(shape_W, index)
        self.b = bias_var(shape_b, index)
        self.vars = {self.W.name: self.W, self.b.name: self.b}

        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope('layer' + str(index)):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weight'):
                variable_summaries(self.W)
                
        self.a_in = a_in
        
        self.kernel_depth_i = kernel_depth_i
        self.layer_size_out = layer_size_out

        out1 = tf.nn.conv2d(self.a_in, self.W, strides=[1, 1, 1, 1], padding='VALID') + self.b
        
        self.out = tf.nn.softmax(out1)
        self.log_out = tf.nn.log_softmax(out1) # Log and softmax!
        
    def get_log_out(self):
        return self.log_out
        
    def get_output(self):
        return self.out


class DeConvLayer(Layer):
    def __init__(self, a_in, W, shape_out, padding = "SAME"):
        tf.add_to_collection('params', W)
        
        self.a_in = a_in
        self.W = W
        self.shape_out = shape_out
        self.padding = padding
    
    def get_output(self):
        out = nn_ops.conv2d_transpose(self.a_in,
                                      self.W,
                                      self.shape_out,
                                      strides=[1, 1, 1, 1],
                                      padding=self.padding,  # padding="VALID",
                                      )
        
        return out


class ConvLayer(Layer):
    def __init__(self, a_in, W, b):
        self.a_in = a_in
        self.W = W
        self.b = b
    
    def get_output(self):
        out = tf.nn.conv2d(self.a_in, self.W, strides=[1, 1, 1, 1], padding='VALID') + self.b
        out2 = relu(out, alpha=1.0e-2)
        return out2


class ConvLayer1(Layer):
    def __init__(self, a_in, kernel_size, kernel_depth, index):
        self.a_in = a_in

        shape_w_1 = [kernel_size[0], kernel_size[1], 1, kernel_depth[1]]
        shape_w_2 = [1, kernel_depth[0], kernel_depth[1]]

        def variable(shape, name_extra, index):
            stddev = 0.1
            initial = tf.truncated_normal(shape, stddev=stddev, name="init_W_" + str(index))
            var = tf.Variable(initial, name='w_' + str(index) + name_extra)

            return var
        
        w1 = variable(shape_w_1, '_1', index)
        w2 = variable(shape_w_2, '_2', index)
        
        w_inter = [tf.reshape(tf.matmul(tf.reshape(w1[..., i], shape = (-1, 1)), w2[..., i]), shape = [kernel_size[0], kernel_size[1], kernel_depth[0], 1]) for i in range(kernel_depth[1])]

        self.W = tf.concat(w_inter, axis = 3, name='concat')
        
        # # TODO output should be of  kernel_depth[1]
        # self.W = tf.reshape(tf.matmul(tf.reshape(w1, shape=(-1, 1)), w2),
        #                     shape=(kernel_size[0], kernel_size[1], kernel_depth[0], 1))
        
        self.vars = {w1.name: w1, w2.name: w2}


    def get_output(self):
        out1 = tf.nn.conv2d(self.a_in, self.W, strides=[1, 1, 1, 1], padding='SAME')
        out2 = relu(out1, alpha = 0.01)
        return out2
    

class ConvSameLayer(Layer):
    def __init__(self, a_in, W, b, keep_prob):
        self.a_in = a_in
        self.W = W
        self.b = b
        self.keep_prob = keep_prob


    def get_output(self):
        out1 = tf.nn.conv2d(self.a_in, self.W, strides=[1, 1, 1, 1], padding='SAME') + self.b
        # Leaky RELU
        out2 = relu(out1, alpha = 0.01)
        # out2 = tf.sigmoid(out1)
        out2_drop = tf.nn.dropout(out2, self.keep_prob)
        return out2_drop
    
    
def relu(x, alpha = 0., max_value = None):
    '''ReLU.
    alpha: slope of negative section.
    '''
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype= tf.float32),
                             tf.cast(max_value, dtype= tf.float32))
        negative_part = tf.clip_by_value(negative_part, tf.cast(0., dtype= tf.float32),
                             tf.cast(max_value, dtype= tf.float32))
    x -= tf.constant(alpha) * negative_part
    return x

class LastLayer(Layer):
    def __init__(self, a_in, W):
        self.a_in = a_in
        self.W = W
        
    def get_output(self):
        return tf.nn.conv2d(self.a_in, self.W, strides=[1, 1, 1, 1], padding='SAME')
    

# Same as conv layer, but the 3 color channels are separated
# TODO
class ConvLayerRGB(ConvLayer):
    def __init__(self, a_in, W, shannel_out=None):
        self.a_in = a_in
        self.W = W
        # self.shape_out = shannel_out
    
    def get_output(self):
        # Separate the 3 color channels
        # out = tf.placeholder(tf.float32, shape=[None, 2, 2, 3])
        
        # def shape(tensor):
        #     s = tensor.get_shape()
        #     return tuple([s[i].value for i in range(0, len(s))])
        
        # W_test = tf.placeholder(tf.float32, shape=[2, 2])
        #
        # foo = tf.multiply(self.a_in[0, 0:2, 0:2, 0], W_test)
        
        # shape_W = shape(self.W)
        
        shape_out = tf.shape(self.W)
        shape_a = tf.shape(self.a_in)
        
        # print("test {}".format(shape_W))
        
        a_in_flat = tf.reshape(self.a_in, [-1, shape_a[1] * shape_a[2] * shape_a[3]])
        
        # out_list = []
        # for index in range(self.shape_out):
        #     out_list.append(tf.reshape(tf.matmul(a_in_flat[..., index], self.W[index]), (-1, 2, 2, 1)))
        
        out = tf.reshape(tf.matmul(a_in_flat, self.W), (-1, 1, 1, shape_out[3]))
        
        # out_list = []
        #
        # for index in range(3):
        #     # out_list[ :, : , :, index] = \
        #     W_i = self.W[index]
        #     out_list.append(tf.nn.conv2d(self.a_in[..., index:index+1],
        #                                 W_i,
        #                                 strides=[1, 1, 1, 1],
        #                                 padding='VALID'))
        #
        # out = tf.concat(out_list, axis=3)
        # out = self.a_in[:, 0:2, 0:2, :] + self.a_in[:, 1:3, 0:2, :] +self.a_in[:, 1:3, 1:3, :] +self.a_in[:, 0:2, 1:3, :]
        return out


def weight_variable(shape, index, bool_trainable = True):
    #todo xavier init
    stddev = np.sqrt(2/(shape[0]*shape[1]*shape[2] +
                        shape[0]*shape[1]*shape[3]))
    initial = tf.truncated_normal(shape, stddev=stddev, name="init_W_" + str(index))

    var = tf.Variable(initial, name='W_' + str(index), trainable=bool_trainable)
    return var


def bias_var(shape, index, bool_trainable = True):
    stddev = 0
    initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=stddev, name="init_b_" + str(index))

    var = tf.Variable(initial, name='b_' + str(index), trainable=bool_trainable)
    return var


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean, collections=['all'])
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev, collections=['all'])
        tf.summary.scalar('max', tf.reduce_max(var), collections=['all'])
        tf.summary.scalar('min', tf.reduce_min(var), collections=['all'])
        tf.summary.histogram('histogram', var, collections=['all'])
                
        mean_input, var_input = tf.nn.moments(var, axes=(0, 1, 3))
        tf.summary.histogram('histogram_in_mean', mean_input, collections=['all'])
        tf.summary.histogram('histogram_in_std', var_input, collections=['all'])

        mean_input, var_input = tf.nn.moments(var, axes=(0, 1, 2))
        tf.summary.histogram('histogram_out_mean', mean_input, collections=['all'])
        tf.summary.histogram('histogram_out_std', var_input, collections=['all'])


def dropoutLayer(input, keep_prob):
    # todo special dropout does not seem a good idea!
    """
    only do dropout in direction of feature maps and batch
    """
    # noise_shape = [tf.shape(input)[0], 1, 1, tf.shape(input)[3]]
    noise_shape = tf.shape(input)

    
    # keep probability at input is higher
    return tf.nn.dropout(input, keep_prob=(keep_prob + 1.0) / 2, noise_shape=noise_shape)


class Layers1():
    def __init__(self, x, layer_config, keep_prob):
            
        self.layer_size = layer_config.layer_size
        self.kernel_depth = layer_config.kernel_depth
        layer_types = layer_config.layer_types
        kernel_size = layer_config.kernel_size
                             
        self.list_a = [x]
        in_place = x
        
        self.var = {}

        in_place = dropoutLayer(in_place, (keep_prob + 1.0)/2)
        self.list_a.append(in_place)
        
        for index, layer_type in enumerate(layer_types):
            
            with tf.name_scope("layer_{}".format(index)):
            
                # TODO REMOVE only train first layer at the moment
                if index == 0:
                    bool_trainable = True
                else:
                    bool_trainable = True
                
                if layer_type == 'fus_conv':
                    """ Combines the output of the previous layer with the whole input"""
                    
                    shape_W = [kernel_size[index][0], kernel_size[index][1],
                               self.kernel_depth[index] + self.kernel_depth[0], self.kernel_depth[index + 1]]
                    shape_b = [1, 1, 1, self.kernel_depth[index + 1]]
                                    
                    W = weight_variable(shape_W, index)
                    b = bias_var(shape_b, index)
    
                    self.var.update({W.name: W})
                    self.var.update({b.name: b})
                    
                    # stacking of previous layer and input
                    # the cropping is not clean yet
                    input_i = tf.concat([in_place, self.list_a[0][:, 2:-2, 2:-2, :]], axis = 3)
                    layer = ConvLayer(input_i, W, b)
                
                elif layer_type == 'last':
                    shape_W = [kernel_size[index][0], kernel_size[index][1],
                               self.kernel_depth[index], self.kernel_depth[index + 1]]
        
                    W = weight_variable(shape_W, index)
        
                    layer = LastLayer(in_place, W)
        
                    self.var.update({W.name: W})

                elif layer_type == 'conv':
                    shape_W = [kernel_size[index][0], kernel_size[index][1],
                               self.kernel_depth[index], self.kernel_depth[index+1]]
                    shape_b = [1, 1, 1, self.kernel_depth[index + 1]]
                    
                    W = weight_variable(shape_W, index, bool_trainable = bool_trainable)
                    
                    b = bias_var(shape_b, index, bool_trainable = bool_trainable)
                    
                    layer = ConvLayer(in_place, W, b)
    
                    self.var.update({W.name: W})
                    self.var.update({b.name: b})
    
                    # Adding a name scope ensures logical grouping of the layers in the graph.
                    with tf.name_scope('layer' + str(index)):
                        # This Variable will hold the state of the weights for the layer
                        with tf.name_scope('weight'):
                            variable_summaries(W)

                elif layer_type == 'deconv':
                    shape_W = [kernel_size[index][0], kernel_size[index][1],
                               self.kernel_depth[index + 1], self.kernel_depth[index]]
                    
                    W = weight_variable(shape_W, index)
                    
                    shape_out = [tf.shape(in_place)[0], self.layer_size[index + 1][0], self.layer_size[index + 1][1],
                                 self.kernel_depth[index + 1]]
                    
                    layer = DeConvLayer(in_place, W, shape_out)
                    
                    self.var.update({W.name: W})

                elif layer_type == 'convrgb':
                    stddev = 0.01
                    
                    shape = (self.layer_size[index][0] * self.layer_size[index][0] * self.kernel_depth[index],
                             self.layer_size[index + 1][0] * self.layer_size[index + 1][0] * self.kernel_depth[index + 1])
                    
                    W_i_init = tf.truncated_normal(shape, stddev=stddev)
                    
                    W = tf.Variable(W_i_init, name='W_' + str(index))
                    
                    layer = ConvLayerRGB(in_place, W)
                    
                    self.var.update({W.name: W})
                    
                # convolution were the input and output shape stays the same
                elif layer_type == 'convsame':
                    shape_W = [kernel_size[index][0], kernel_size[index][1],
                               self.kernel_depth[index], self.kernel_depth[index + 1]]
    
                    shape_b = [1, 1, 1, self.kernel_depth[index + 1]]
    
                    W = weight_variable(shape_W, index)
                    
                    b = bias_var(shape_b, index)
    
                    layer = ConvSameLayer(in_place, W, b, keep_prob)
    
                    self.var.update({W.name: W})
                    self.var.update({b.name: b})
    
                    # Adding a name scope ensures logical grouping of the layers in the graph.
                    with tf.name_scope('layer' + str(index)):
                        # This Variable will hold the state of the weights for the layer
                        with tf.name_scope('weight'):
                            variable_summaries(W)

                elif layer_type == 'conv1':
       
                    layer = ConvLayer1(in_place, kernel_size[index],
                                       self.kernel_depth[index: index+2], index)
                    
                    self.var.update(layer.vars)

                elif layer_type == 'convsameflat':
                    shape_W = [1, kernel_size[index][1],
                               self.kernel_depth[index], self.kernel_depth[index + 1]]
        
                    shape_b = [1, 1, 1, self.kernel_depth[index + 1]]
        
                    W = weight_variable(shape_W, index)
        
                    b = bias_var(shape_b, index)
        
                    layer = ConvSameLayer(in_place, W, b, keep_prob)
        
                    self.var.update({W.name: W})
                    self.var.update({b.name: b})

                elif layer_type == 'deconvsame':
                    shape_W = [kernel_size[index][0], kernel_size[index][1],
                               self.kernel_depth[index + 1], self.kernel_depth[index]]
    
                    W = weight_variable(shape_W, index)
    
                    shape_out = [tf.shape(in_place)[0], self.layer_size[index + 1][0], self.layer_size[index + 1][1],
                                 self.kernel_depth[index + 1]]
    
                    layer = DeConvLayer(in_place, W, shape_out, padding = "SAME")
    
                    self.var.update({W.name: W})

                elif layer_type == 'softmax':
                    kernel_size_i = kernel_size[index]
                    kernel_depth_i = self.kernel_depth[index : index+2]
                    layer_size_out = self.layer_size[index + 1]
                    
                    layer = SoftmaxLayer(in_place, kernel_size_i,
                                         kernel_depth_i, layer_size_out, index)
                    self.var.update(layer.vars)
                    
                    self.log_out = layer.get_log_out()

                elif layer_type == 'softmaxsconv':
                    kernel_size_i = kernel_size[index]
                    kernel_depth_i = self.kernel_depth[index: index + 2]
                    layer_size_out = self.layer_size[index + 1]
        
                    layer = SoftmaxLayerSame(in_place, kernel_size_i,
                                         kernel_depth_i, layer_size_out, index)
                    self.var.update(layer.vars)
        
                    self.log_out = layer.get_log_out()
                    
                else:
                    errmsg = "Unknown layer type: " + layer_type
                    raise (ValueError(errmsg))
    
                in_place = layer.get_output()
                self.list_a.append(in_place)
        
                # dropout
                if layer_type == 'conv':
                    in_place = dropoutLayer(in_place, keep_prob)
                    self.list_a.append(in_place)


    # For training on subset of NN
    def get_res(self, depth = -2):
        a_depth = self.list_a[depth]
        
        width = self.layer_size[depth][0]
        kernel_depth = self.kernel_depth[depth]
        shape_W = [width, width, kernel_depth, 1]

        stddev = 0.01
        name = "res_W_" + str(depth)
        initial = tf.truncated_normal(shape_W, stddev=stddev, name="init_" + name)

        W = tf.Variable(initial, name=name)
        self.var.update({W.name: W})
        
        layer = LastLayer(a_depth, W)
        
        # residue is this output + x
        return layer.get_output() + self.list_a[0]
    
config1 = tf.ConfigProto(
                         log_device_placement=True,
                         allow_soft_placement=True
                         )
config1.gpu_options.per_process_gpu_memory_fraction = 0.8

class NetworkGroup(object):
    def __init__(self, layers, bool_residue = False):
        self.sess = tf.Session(config=config1)
        
        self.x = tf.placeholder(tf.float32, shape= layers.x_shape(), name='x')
        self.x_tophat = tf.placeholder(tf.float32, shape=layers.x_shape(), name='x_tophat')
        self.y = tf.placeholder(tf.float32, shape= layers.y_shape(), name='y')
        self.params = {}

        # for dropout
        self.keep_prob = tf.placeholder_with_default(input = 1.0, shape=( ), name='keep_prob')

        if bool_residue:
            self.a_layers = Layers1(self.x_tophat, layers, self.keep_prob)  # network.Layers1(self.x, layer_config)

        else:
            self.a_layers = Layers1(self.x, layers, self.keep_prob)  # network.Layers1(self.x, layer_config)
    
        self.update_params()
    
        if bool_residue:
            # TODO
            self.ff = self.a_layers.list_a[-1] + self.x[:, 7:7+32, 7:7+32, :]  # residue
        else:
            self.ff = self.a_layers.list_a[-1]

        self.global_epoch = tf.Variable(0, name='global_epoch',
                                   trainable=False)  # counts full training steps (all mini batches)
        self.params.update({self.global_epoch.name: self.global_epoch})
                
        if layers.cost == 'l2':
            # training costs
            diff = tf.subtract(self.y, self.ff)
            shape = tf.shape(diff)
            cost_i = tf.reshape(tf.square(diff), [-1, shape[1] * shape[2] * shape[3]])
            cost = tf.reduce_mean(cost_i, [-1])
        elif layers.cost == 'xentropy':
            cost = -tf.reduce_mean(self.y * self.a_layers.log_out)
        elif layers.cost =='wxentropy':
            a = tf.multiply(self.y * self.a_layers.log_out, layers.w_c)
            cost = -tf.reduce_mean(a)
            # cost = -tf.reduce_mean(layers.w_c * self.y * self.a_layers.log_out)
        else:
            msg = """Error choosing cost function. Check name."""
            raise ValueError(msg)

        # regularizer1 = 0
        regularizer2 = 0

        for param in self.params:
            if param[0] == 'W':
                # regularizer1 = regularizer1 + tflearn.losses.L1(self.params[param])
                regularizer2 = regularizer2 + tflearn.losses.L2(self.params[param])


        # beta1 = 1.0e-5
        beta2 = 1.0e-2
        
        self.cost_mean_list = (tf.reduce_mean(cost), beta2*regularizer2)
        self.cost_mean = self.cost_mean_list[0] + self.cost_mean_list[1]

        # def variable_summaries(var):
        #     """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        #     with tf.name_scope('summaries'):
        #         mean = tf.reduce_mean(var)
                
        #         with tf.name_scope('stddev'):
        #             stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        #         tf.summary.scalar('stddev', stddev)
        #         tf.summary.scalar('max', tf.reduce_max(var))
        #         tf.summary.scalar('min', tf.reduce_min(var))
        #         tf.summary.histogram('histogram', var)
        #
        with tf.name_scope('cost'):
            mean = tf.reduce_mean(self.cost_mean)
            tf.summary.scalar('mean', mean, collections=['all','test', 'cost'])
            tf.summary.scalar('acc_cost', tf.reduce_mean(self.cost_mean_list[0]), collections=['all', 'cost'])
            tf.summary.scalar('l2_cost', tf.reduce_mean(self.cost_mean_list[1]), collections=['all', 'cost'])
           
            if bool_residue:
                # With values floats between 0 and 1!
                psnr = -10.0*tf.log(mean)/np.log(10.0) # no log10
                tf.summary.scalar('psnr', psnr, collections=['all', 'cost'])
            
        self.train_dep = {}
        self.cost_dep = {}
        
        self.update_saver()
        
        self.__set_ops()
        
    def pruning(self):
        
        w = self.params['W_7:0']
        b7 = self.params['b_7:0']
        w8 = self.params['W_8:0']

        # mean, var = tf.nn.moments(w, axes=[0, 1, 2])
        
        # arg = tf.argmin(var)
        # w[: ,: ,: , arg] = 0
        
        w1 = self.sess.run(w)
        b71 = self.sess.run(b7)
        w81 = self.sess.run(w8)

        var = np.std(w1, axis=(0, 1, 2))
        arg = np.argsort(var)
        w1 = w1[:, :, :, arg[-100:]]
        
        b71 = b71[:,  :, : , arg[-100:]]
        w81 = w81[:, :, arg[-100:], :]
        
        
        # Make new weights
        shape_W = list(np.shape(w1))
        shape_W[3] = 100
        W_star7 = weight_variable(shape_W, 7)
        
        # shape_b = list(tf.shape(b7))
        # b_star7 = bias_var(shape_b, 7)
        # shape_W = list(tf.shape(w8))
        # W_star8 = weight_variable(shape_W, 8)
        
        # tf.add_to_collection('params', W_star7)
        
        # w = W_star7

        self.params['W_7:0'] = W_star7
        
        w = self.params['W_7:0']
        
        # print(W_star7.name)
        
        # self.params.update({W_star7.name: W_star7})

        # w = self.params['W_7:0']
        self.sess.run(w.assign(w1))
        # self.sess.run(w8.assign(w81))
        # self.sess.run(b7.assign(b71))

        
        # tf.scatter_update(w, )
           
        
        for param in self.params:
            print(param)
            
            
            
            # if param[0] == "W" or param[0:5] == "res_W":

    def __set_ops(self):
        self.__op_plus_global_epoch = tf.assign(self.global_epoch, self.global_epoch + 1)
        
    def remove_nan(self):
        # if w is nan use 1 * NUMBER else use element in w

        print('\n')
        for param in self.params:
            if param[0] == "W" or param[0:5] == "res_W":
                print(param)
                w = self.params[param]
                NUMBER = 0.1
                w = tf.where(tf.is_nan(w), tf.ones_like(w, dtype= tf.float32) * NUMBER, w)
                w = tf.where(tf.greater(tf.abs(w), 1e0), tf.ones_like(w, dtype=tf.float32) * NUMBER, w)
                w = tf.where(tf.greater(1e-10, tf.abs(w)), tf.ones_like(w, dtype=tf.float32) * NUMBER, w)
                
                w = tf.ones_like(w, dtype= tf.float32)
                self.params[param] = w
                     
    def update_params(self):
        self.params.update(self.a_layers.var)

    def set_lr(self, lr, mom = 0.9):
        self.optimizer = tf.train.AdamOptimizer(lr, beta1=mom)
        
    @lam_warnings.deprecated
    def set_gradient_clipping(self, clip_val):
        warnings.warn("Use set_train(self, clipping_val = clip_val)", category=DeprecationWarning)
        
    def set_train(self, bool_clipping = False):
        """
        :param clipping_val: A float value to limit the gradients
        :return:
        """
        # # Calculate gradients
        # gvs = self.optimizer.compute_gradients(self.cost_mean)
        #
        # # Process gradients
        
        gvs = self.optimizer.compute_gradients(self.cost_mean)
        
        if bool_clipping:
            grads, vars = zip(*gvs)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            gvs = zip(grads, vars)
        else:
            ...
            
        self.train_op = self.optimizer.apply_gradients(gvs)

        # self.gvs_mean1 = [tf.reduce_mean(tf.abs(grad)) for grad, var in tuple(gvs)]
        

    def train(self, feed_dict):
        """ feed_dict should contain x and y"""
        self.sess.run(self.train_op, feed_dict=feed_dict)
        # # TODO
        # gvs_mean = self.sess.run(self.gvs_mean1, feed_dict=feed_dict)
        # print(gvs_mean)
        
    def train_depth(self, feed_dict, depth = -2):
        train_op = self.train_dep[depth]
        self.sess.run(train_op, feed_dict=feed_dict)
        
    def cost(self, feed_dict):
        return self.sess.run(self.cost_mean_list, feed_dict = feed_dict)
        
    def cost_depth(self, feed_dict, depth = -2):
        cost_i = self.cost_dep[depth]
        return self.sess.run(cost_i, feed_dict = feed_dict)
    
    def get_output(self, feed_dict):
        """ Only x is needed """
        return self.sess.run(self.ff, feed_dict=feed_dict)

    def close_session(self):
        self.sess.close()
       
    def update_saver(self):
        self.saver = tf.train.Saver(self.params)
       
    def load_params(self, path):
        # # TODO remove
        # params = self.params
        #
        # print('\n\n\n\n\n')
        # print(params)
        # print('\n\n\n\n\n')
        #
        # params = {param_name: params[param_name] for param_name in params if param_name[2] is not '1'}
        #
        # print('\n\n\n\n\n')
        # print(params)
        # print('\n\n\n\n\n')
        # self.saver = tf.train.Saver(params)
        # self.saver.restore(self.sess, path)
        #
        # self.update_saver()

        # self.saver = tf.train.Saver(self.params)


        ckpt = tf.train.get_checkpoint_state(path)
        print(ckpt.model_checkpoint_path)
        
        # self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        
        for param_i in self.params:
            print(param_i)
        
        params2 = {'W_9:0': self.params['layer_9/W_9:0']}
        saver2 = tf.train.Saver(params2)
        saver2.restore(self.sess, ckpt.model_checkpoint_path)
        
     
        
        #self.train.get_checkpoint
        
    def save_params(self, dir):
        """ In order to make sure all saves are similar"""
        print('-- saving --')
        self.saver.save(self.sess, dir + 'chpt', global_step=self.get_global_epoch(), write_meta_graph=False)

    # TODO
    def set_train_dep(self, lr, depth = -2):
        diff = tf.subtract(self.y, self.a_layers.get_res(depth))
        
        # Update the additional weight
        self.update_params()
        
        shape = tf.shape(diff)
        cost_i = tf.reshape(tf.square(diff), [-1, shape[1] * shape[2] * shape[3]])
        cost = tf.reduce_mean(cost_i, [-1])
        cost_mean = tf.reduce_mean(cost)
        train_op = tf.train.AdamOptimizer(lr).minimize(cost_mean)
        
        dict = {depth: train_op}
        self.train_dep.update(dict)
        dict_cost = {depth: cost_mean}
        self.cost_dep.update(dict_cost)

    def set_summary(self, dir):
        self.merged = tf.summary.merge_all('all')
        self.merged_test = tf.summary.merge_all('test')
        self.__summ_set_cost = tf.summary.merge_all('cost')
        
        self.train_writer = tf.summary.FileWriter(dir + 'train', flush_secs = 20)
        self.test_writer = tf.summary.FileWriter(dir + 'test', flush_secs = 20)
        # save the graph only once each run
        self.graph_writer = tf.summary.FileWriter(dir + 'graph', self.sess.graph)
        
        self.__summary_writer = {'data1' : tf.summary.FileWriter(dir + 'data1', flush_secs=20),
                               'data2' : tf.summary.FileWriter(dir + 'data2', flush_secs=20)}
                
    def get_global_epoch(self):
        return self.sess.run(self.global_epoch)
    
    def plus_global_epoch(self):
        return self.sess.run(self.__op_plus_global_epoch)
        
    def update_summary(self, summary):
        global_epoch = self.get_global_epoch()
        print(global_epoch)
        self.train_writer.add_summary(summary, global_epoch)
        
    def update_summary_test(self, summary):
        global_epoch = self.get_global_epoch()
        self.test_writer.add_summary(summary, global_epoch)
        
    def update_summary_i(self , name_writer, feed_dict):
        """ name_writer: Name of data
        Name of values to save"""

        summary = self.sess.run(self.__summ_set_cost, feed_dict = feed_dict)
        global_epoch = self.get_global_epoch()
        self.__summary_writer[name_writer].add_summary(summary, global_epoch)
               
    def set_summary_conf(self, dir, conf):
        # self.conf1_writer = tf.summary.FileWriter(dir + 'TP')
        #
        # a = tf.summary.scalar('TP', conf[0, 0])
        # self.summ1 = tf.summary.merge(a)
        # TODO
        ...
        
    def update_summary_conf(self, summary):
        # global_epoch = self.get_global_epoch()
        #
        # self.conf1_writer.add_summary(summary ,global_epoch)
        # self.conf1_writer.add_summary(vector[0, 0], global_epoch)
        # self.conf1_writer.add_summary(vector[0, 0], global_epoch)
        # self.conf1_writer.add_summary(vector[0, 0], global_epoch)
        # TODO
        ...

    def load_init(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)
 