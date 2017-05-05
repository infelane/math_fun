import keras.models
import keras.backend as K

try:
    import h5py
except ImportError:
    h5py = None


class Sequential(keras.models.Sequential):
    def __init__(self, layers=None, name=None):
        super(Sequential, self).__init__(layers, name)
    
    # TODO

    def load_weights(self, filepath, by_name=False):
        # TODO
        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']
            
        print([a for a in f.attrs])
        
        # print(f['model_weights'])
        print(f.attrs['layer_names'])
        
        layers = self.layers

        """Implements topological (order-based) weight loading.

        # Arguments
            f: A pointer to a HDF5 group.
            layers: a list of target layers.

        # Raises
            ValueError: in case of mismatch between provided layers
                and weights file.
        """
        if 'keras_version' in f.attrs:
            original_keras_version = f.attrs['keras_version'].decode('utf8')
        else:
            original_keras_version = '1'
        if 'backend' in f.attrs:
            original_backend = f.attrs['backend'].decode('utf8')
        else:
            original_backend = None

        filtered_layers = []
        for layer in layers:
            weights = layer.weights
            if weights:
                filtered_layers.append(layer)
                
        from keras.engine.topology import preprocess_weights_for_loading

        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
        filtered_layer_names = []
        for name in layer_names:
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            if weight_names:
                filtered_layer_names.append(name)
        layer_names = filtered_layer_names
        if len(layer_names) != len(filtered_layers):
            raise ValueError('You are trying to load a weight file '
                             'containing ' + str(len(layer_names)) +
                             ' layers into a model with ' +
                             str(len(filtered_layers)) + ' layers.')

        # We batch weight value assignments in a single backend call
        # which provides a speedup in TensorFlow.
        weight_value_tuples = []
        for k, name in enumerate(layer_names):
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            weight_values = [g[weight_name] for weight_name in weight_names]
            layer = filtered_layers[k]
            symbolic_weights = layer.weights
            weight_values = preprocess_weights_for_loading(layer,
                                                           weight_values,
                                                           original_keras_version,
                                                           original_backend)
            if len(weight_values) != len(symbolic_weights):
                raise ValueError('Layer #' + str(k) +
                                 ' (named "' + layer.name +
                                 '" in the current model) was found to '
                                 'correspond to layer ' + name +
                                 ' in the save file. '
                                 'However the new layer ' + layer.name +
                                 ' expects ' + str(len(symbolic_weights)) +
                                 ' weights, but the saved weights have ' +
                                 str(len(weight_values)) +
                                 ' elements.')
            weight_value_tuples += zip(symbolic_weights, weight_values)
        
        K.batch_set_value(weight_value_tuples)
        
        f.close()
        # raise
    
        # # Legacy support
        # if legacy_models.needs_legacy_support(self):
        #     layers = legacy_models.legacy_sequential_layers(self)
        # else:
        #     layers = self.layers
        # if by_name:
        #     topology.load_weights_from_hdf5_group_by_name(f, layers)
        # else:
        #     topology.load_weights_from_hdf5_group(f, layers)
        # if hasattr(f, 'close'):
        #     f.close()