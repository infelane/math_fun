# extends keras's metrics

from keras import backend as K
import numpy as np

import tensorflow as tf


def sum_and(a, b):
    return K.sum(a * b)  # np.logical_and(a, b) )

def accuracy_test(y_true, y_pred):
    arg_true = K.argmax(y_true, axis=-1)
    arg_pred = K.argmax(y_pred, axis=-1)
    
    true0 = K.cast(K.equal(arg_true, 0), np.float32)
    pred0 = K.cast(K.equal(arg_pred, 0), np.float32)
    true1 = K.cast(K.equal(arg_true, 1), np.float32)
    pred1 = K.cast(K.equal(arg_pred, 1), np.float32)
    
    return true0, pred0, true1, pred1
    
def TP(y_true, y_pred):
    true0, pred0, true1, pred1 = accuracy_test(y_true, y_pred)
    return sum_and(true0, pred0)

def FP(y_true, y_pred):
    true0, pred0, true1, pred1 = accuracy_test(y_true, y_pred)
    return sum_and(true1, pred0)

def FN(y_true, y_pred):
    true0, pred0, true1, pred1 = accuracy_test(y_true, y_pred)
    return sum_and(true0, pred1)

def TN(y_true, y_pred):
    true0, pred0, true1, pred1 = accuracy_test(y_true, y_pred)
    return sum_and(true1, pred1)

def sens(y_true, y_pred):
    return TN(y_true, y_pred) / (TN(y_true, y_pred) + FP(y_true, y_pred))

def prec(y_true, y_pred):
    return TN(y_true, y_pred) / (TN(y_true, y_pred) + FN(y_true, y_pred))
    
    
    
