# Extends keras.losses

import numpy as np
import lambnet.tensorflow_backend as K  # TODO also for theano


def weigthed_crossentropy(weight = [1, 1]):
    '''
    like crossentropy, but also
    :param weight: as list of the weights
    :return: the cost function to be given with keras' compile
    '''
    weight = np.asarray(weight)

    return lambda y_true, y_pred : K.weighted_categorical_crossentropy(y_true, y_pred, weight)
