"""
constantly reused network structures
"""

from keras.layers import Input

def gen_in(w:int, ext:int or tuple, depth:int, name):
    if type(ext) is tuple:
        shape = (w + ext[0] + ext[1], w + ext[0] + ext[1], depth)
    if type(ext) is int:
        shape = (w + 2 * ext, w + 2 * ext, depth)
    return Input(shape=shape, name=name)
