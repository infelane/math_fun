import tensorflow as tf

import pickle


class Layers(object):
    def __init__(self, layer_size, kernel_size, kernel_depth):
        self.params = {}

        # The None, is the batchsize that can be of any size
        self.x = tf.placeholder(tf.float32, shape=[None, layer_size[0][0], layer_size[0][1], kernel_depth[0]])
        # Target output classes
        # TODO accept output patch
        # self.y = tf.placeholder(tf.float32, shape=[None, layer_size[-1][0], layer_size[-1][0], kernel_depth[-1]])
        self.y = tf.placeholder(tf.float32, shape=[None, kernel_depth[-1]])

        self.layer_values = []

        def multiply(numbers):
            total = 1
            for x in numbers:
                total *= x
            return total

        def weight_variable(shape, index):

            stddev = 1.0/(multiply(shape[:-1]))
            initial = tf.truncated_normal(shape, stddev=stddev, name = "init_W_" + str(index))
            var = tf.Variable(initial, name='W_' + str(index))
            self.params.update({var.name: var})
            return var

        def bias_variable(shape, index):
            stddev = 0.01
            initial = tf.truncated_normal(shape=shape, stddev=stddev, name = "init_b_" + str(index))
            var = tf.Variable(initial, name='b_' + str(index))
            self.params.update({var.name: var})
            return var

        a_layers = [self.x]
        i = 0
        for i, kernel in enumerate(kernel_size[:-1]):
            W = weight_variable([kernel[0], kernel[1], kernel_depth[i], kernel_depth[i+1]], i)
            b = bias_variable([kernel_depth[i+1]], i)

            layer_val = Layer_val()
            layer_val.W = W
            layer_val.b = b
            self.layer_values.append(layer_val)

            # a_layers.append(self.conv(a_layers[-1], W, b))
            layer = ConvLayer(W, b)
            layer.set_input(a_layers[-1])
            a_layers.append(layer.get_output())

        #last layer fully connected
        kernel = kernel_size[-1]
        W = weight_variable([kernel[0] * kernel[1] * kernel_depth[-2], kernel_depth[-1]], i+1)
        b = bias_variable([kernel_depth[-1]], i+1)
        self.params.update({W.name: W})
        self.params.update({b.name: b})
        tf.add_to_collection('params', W)
        tf.add_to_collection('params', b)
        for_last_layer = tf.reshape(a_layers[-1], (-1, kernel[0] * kernel[1] * kernel_depth[-2]))
        a_layers.append(tf.matmul(for_last_layer, W) + b)

        self.ff = a_layers[-1]

    def get_ff(self):
        return self.ff

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def conv_pool(self, x, W, b):
        tf.add_to_collection('params', W)
        tf.add_to_collection('params', b)
        z = tf.nn.relu(self.conv2d(x, W) + b)
        return self.max_pool_2x2(z)

    def conv(self, x, W, b):
        tf.add_to_collection('params', W)
        tf.add_to_collection('params', b)
        return tf.nn.relu(self.conv2d(x, W) + b)


#Abstract class
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

class ConvPoolLayer(Layer):
    ...

class ConvLayer(Layer):
    def __init__(self, W, b):
        tf.add_to_collection('params', W)
        tf.add_to_collection('params', b)

        self.W = W
        self.b = b

    def set_input(self, inp):
        self.inp = inp

    def get_output(self):
        out = tf.nn.relu(conv2d(self.inp, self.W) + self.b)
        return out


def conv2d(x, W):
    # padding='VALID' reduced output size, 'SAME' keeps the same output size
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

class Layer_val():
    def __init__(self):
        self.w = None
        self.b = None
