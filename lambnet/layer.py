# -*- coding: utf-8 -*-
from __future__ import absolute_import

from keras import backend as K
from keras.engine import Layer, InputSpec

# imports for backwards namespace compatibility


class LayerConfig(object):
    def __init__(self, layer_size = [[1, 1], [1, 1]],
                 kernel_size = [1, 1],
                 kernel_depth = [1],
                 layer_types = ['conv'],
                 layer_size_in = [1, 1],
                 layer_size_out=[1, 1],
                 cost=None,
                 k=None,
                 r=None):
        self.layer_size_in = layer_size_in
        self.layer_size_out = layer_size_out
        # self.layer_size = layer_size
        self.kernel_size = kernel_size
        self.kernel_depth = kernel_depth
        self.layer_types = layer_types
        if cost == 'xentropy':
            self.cost = cost
        elif cost == 'wxentropy':
            self.cost = cost
            self.w_c = ((r + 1) / (r + k), k * (r + 1) / (r + k))
        else:
            self.cost = 'l2'
    
    def x_shape(self):
        return [None, self.layer_size[0][0], self.layer_size[0][1], self.kernel_depth[0]]
    
    def y_shape(self):
        return [None, self.layer_size[-1][0], self.layer_size[-1][1], self.kernel_depth[-1]]
    

class Cropping2D(Layer):
    '''Cropping layer for 2D input (e.g. picture).

    # Input shape
        4D tensor with shape:
        (samples, depth, first_axis_to_crop, second_axis_to_crop)

    # Output shape
        4D tensor with shape:
        (samples, depth, first_cropped_axis, second_cropped_axis)

    # Arguments
        padding: tuple of tuple of int (length 2)
            How many should be trimmed off at the beginning and end of
            the 2 padding dimensions (axis 3 and 4).
    '''
    input_ndim = 4

    def __init__(self, cropping=((1,1),(1,1)), dim_ordering=K.image_dim_ordering(), **kwargs):
        super(Cropping2D, self).__init__(**kwargs)
        assert len(cropping) == 2, 'cropping mus be two tuples, e.g. ((1,1),(1,1))'
        assert len(cropping[0]) == 2, 'cropping[0] should be a tuple'
        assert len(cropping[1]) == 2, 'cropping[1] should be a tuple'
        self.cropping = tuple(cropping)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=4)]

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':

            return (input_shape[0],
                    input_shape[1],
                    input_shape[2] - self.cropping[0][0] - self.cropping[0][1],
                    input_shape[3] - self.cropping[1][0] - self.cropping[1][1])
        elif self.dim_ordering == 'tf':
            return (input_shape[0],
                    input_shape[1] - self.cropping[0][0] - self.cropping[0][1],
                    input_shape[2] - self.cropping[1][0] - self.cropping[1][1],
                    input_shape[3])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        """
        width, height = self.output_shape()[2], self.output_shape()[3]
        width_crop_left = self.cropping[0][0]
        height_crop_top = self.cropping[1][0]
        
        return x[:, :, width_crop_left:width+width_crop_left, height_crop_top:height+height_crop_top]
        """
        if self.dim_ordering == 'th':
            return x[:, :, self.cropping[0][0]:-self.cropping[0][1], self.cropping[1][0]:-self.cropping[1][1]]
        
        elif self.dim_ordering == 'tf':
            return x[:, self.cropping[0][0]:-self.cropping[0][1], self.cropping[1][0]:-self.cropping[1][1], :]

        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        

    def get_config(self):
        config = {'cropping': self.padding}
        base_config = super(Cropping2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
