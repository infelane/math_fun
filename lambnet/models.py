import keras.models
import keras.backend as K
import numpy as np

try:
    import h5py
except ImportError:
    h5py = None


class Sequential(keras.models.Sequential):
    def __init__(self, layers=None, name=None):
        super(Sequential, self).__init__(layers, name)
        
    # # don't think we need this
    # def save_weights(self, filepath, layer_i = [0, 1]):
    #
    #     if h5py is None:
    #         raise ImportError('`save_weights` requires h5py.')
    #     # If file exists and should not be overwritten:
    #
    #     layers = self.layers[layer_i[0] : layer_i[-1]]
    #
    #     f = h5py.File(filepath, 'w')
    #     save_weights_to_hdf5_group(f, layers)
    #     f.flush()
    #     f.close()

    def load_weights(self, filepath, layer_i = [0, 1]):
        """
        Loads the weights layer per layer, can be easily configered to only load certain layers
        :param filepath:
        :param layer_i: # TODO
        :return:
        """
        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']
            
        index_l = np.arange(0, 3)   # FILE layer numbering
        index_f = np.arange(0, 3)   # MODEL layer numbering
            
        for f_i, l_i in zip(index_f, index_l):
            load_weights_hdf5_i(f, self.layers, f_i, l_i)
        
        if hasattr(f, 'close'):
            f.close()
        
    
# def save_weights_to_hdf5_group(f, layers):
#     from keras import __version__ as keras_version
#
#     f.attrs['layer_names'] = [layer.name.encode('utf8') for layer in layers]
#     f.attrs['backend'] = K.backend().encode('utf8')
#     f.attrs['keras_version'] = str(keras_version).encode('utf8')
#
#     for layer in layers:
#         g = f.create_group(layer.name)
#         symbolic_weights = layer.weights
#         weight_values = K.batch_get_value(symbolic_weights)
#         weight_names = []
#         for i, (w, val) in enumerate(zip(symbolic_weights, weight_values)):
#             if hasattr(w, 'name') and w.name:
#                 name = str(w.name)
#             else:
#                 name = 'param_' + str(i)
#             weight_names.append(name.encode('utf8'))
#         g.attrs['weight_names'] = weight_names
#         for name, val in zip(weight_names, weight_values):
#             param_dset = g.create_dataset(name, val.shape,
#                                           dtype=val.dtype)
#             if not val.shape:
#                 # scalar
#                 param_dset[()] = val
#             else:
#                 param_dset[:] = val
      
def load_weights_hdf5_i(f, layers, f_i, l_i = None):
    # The model and save file correspond
    if not l_i:
        l_i = f_i

    layer = layers[l_i]
    weights_l = layer.weights
    
    name_f = (f.attrs['layer_names'][f_i]).decode('utf8')
    g = f[name_f]
    weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
    
    print('model layer name: {}'.format(layer.name))
    print('h5 layer name: {}'.format(name_f))
    
    
    if len(weight_names) != len(weights_l):
        raise ValueError('Layer #' + str(l_i) +
                         ' (named "' + layer.name +
                         '" in the current model) was found to '
                         'correspond to layer ' + name_f +
                         ' in the save file. '
                         'However the new layer ' + layer.name +
                         ' expects ' + str(len(weights_l)) +
                         ' weights, but the saved weights have ' +
                         str(len(weight_names)) +
                         ' elements.')

    print('---------------------------------------------')
    
    if len(weights_l) == 0:
        # this layer has no weights
        return
        
    weight_f = [g[weight_name] for weight_name in weight_names]
    weight_value_tuples = zip(weights_l, weight_f)
    K.batch_set_value(weight_value_tuples)
    
 
    