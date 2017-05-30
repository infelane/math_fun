import numpy as np
from tensorflow.python.framework import dtypes
from six.moves import xrange  # pylint: disable=redefined-builtin
import sys
import os
import pickle

# cmd_subfolder = os.path.realpath(
# 	os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "subfolder")))
# sys.path.append( <path to dirFoo> )
folder_loc = '/ipi/private/lameeus/private_Documents/python/2016_November/PhD/packages'
cmd_subfolder = os.path.realpath(folder_loc)
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
from training_data import Training_data

import data_net
        

folder_loc = '/ipi/private/lameeus/private_Documents/python/2017_February/super_res_challenge'
cmd_subfolder = os.path.realpath(folder_loc)
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import data_net
# from training_data import Training_data
from PIL import Image
import matplotlib.pyplot as plt
folder_loc = '/ipi/private/lameeus/private_Documents/python/2017_04'
cmd_subfolder = os.path.realpath(folder_loc)
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import block_data

  
# todo return test data
def ground_truth(width = 16, ext = 0):
    
    a = block_data.data_train(width, ext)

    # todo validation data
    b = block_data.data_valid(width, ext)
    c = block_data.data_test(width, ext)
    
    return (a, b, c)


def data_ex2(new = True, ext_in = 1, ext_out = 0, n_i = 10000):
    folder = '/scratch/lameeus/data/tensorflow/io/'
    
    name_tr = "data_tr2.p"
    name_va = "data_va2.p"
    name_te = "data_te2.p"

    if new:
        n = (n_i, n_i, n_i)
        images_set = 'beard'

        training_data = Training_data(new=True, amount=sum(n), ext_in=ext_in,
                                      ext_out=ext_out, images_set=images_set)
        
        training_data, validation_data, test_data = training_data.split_theano(n=n)
        
        in_tr = (training_data[0].eval() + 0.5)*255
        out_tr = training_data[1].eval()
        in_va = (validation_data[0].eval() + 0.5)*255
        out_va = validation_data[1].eval()
        in_te = (test_data[0].eval() + 0.5)*255
        out_te = test_data[1].eval()

        data_tr = DataSet(data_reshape(in_tr, ext_in, 7), out_reshape(out_tr, ext_out), depth=True, reshape=False)
        data_va = DataSet(data_reshape(in_va, ext_in, 7), out_reshape(out_va, ext_out), depth=True, reshape=False)
        data_te = DataSet(data_reshape(in_te, ext_in, 7), out_reshape(out_te, ext_out), depth=True, reshape=False)

        pickle.dump(data_tr, open(folder + name_tr, "wb"))
        pickle.dump(data_va, open(folder + name_va, "wb"))
        pickle.dump(data_te, open(folder + name_te, "wb"))
    else:
        data_tr = pickle.load(open(folder + name_tr, "rb"))
        data_va = pickle.load(open(folder + name_va, "rb"))
        data_te = pickle.load(open(folder + name_te, "rb"))
        
    return data_tr, data_va, data_te


def data_reshape(inp, ext, depth):
    return inp.reshape((-1, ext * 2 + 1, ext * 2 + 1, depth))


def out_reshape(out, ext):
    out_new = np.zeros((np.shape(out)[0] * np.shape(out)[1], 2))
    out = np.array(out).reshape((-1,))
    for index, out_index in enumerate(out):
        out_new[index, out_index] = 1
    
    return np.reshape(out_new, newshape=(-1, ext*2+1, ext*2+1, 2))
        

def data_ex1(new=True):
    folder = '/scratch/lameeus/data/tensorflow/io/'
    if new:
        n_train = 10000
        n = (n_train, 10000, 10000)
        images_set = 'beard'
        ext_in = 1
        ext_out = 0
        training_data = Training_data(new=True, amount=sum(n), ext_in=ext_in,
                                      ext_out=ext_out, images_set=images_set)

        training_data_gpu, validation_data_gpu, test_data_gpu = training_data.split_theano(n=n)

        in_tr = (training_data_gpu[0].eval() + 0.5)*255
        out_tr = training_data_gpu[1].eval()
        in_va = (validation_data_gpu[0].eval() + 0.5)*255
        out_va = validation_data_gpu[1].eval()
        in_te = (test_data_gpu[0].eval() + 0.5)*255
        out_te = test_data_gpu[1].eval()

        data_tr = DataSet(data_reshape(in_tr, ext_in), out_reshape(out_tr), depth=True, reshape=False)
        data_va = DataSet(data_reshape(in_va, ext_in), out_reshape(out_va), depth=True, reshape=False)
        data_te = DataSet(data_reshape(in_te, ext_in), out_reshape(out_te), depth=True, reshape=False)

        pickle.dump(data_tr, open(folder + "data_tr.p", "wb"))
        pickle.dump(data_va, open(folder + "data_va.p", "wb"))
        pickle.dump(data_te, open(folder + "data_te.p", "wb"))
    else:
        data_tr = pickle.load(open(folder + "data_tr.p", "rb"))
        data_va = pickle.load(open(folder + "data_va.p", "rb"))
        data_te = pickle.load(open(folder + "data_te.p", "rb"))

    return data_tr, data_va, data_te
