import numpy as np
import keras.backend as K


def categorical_accuracy(y_true, y_pred):
    """ takes into account that y_pred can be 0 (no annotation)"""
    
    bool_annot = np.not_equal(np.sum(y_true, axis=-1), 0)

    y_true_annot = y_true[bool_annot, :]
    y_pred_annot = y_pred[bool_annot, :]
    
    return np.mean(np.equal(np.argmax(y_true_annot, axis=-1),
                            np.argmax(y_pred_annot, axis=-1)))


def metrics_after_predict(metric_func, args):
    """ Metrics are evaluated in keras backend domain
    Thus 1) convert to backend
    2) evaluate back to original domain
    """
    args_k = [K.variable(arg_i) for arg_i in args]
    return K.eval(metric_func(*args_k))
