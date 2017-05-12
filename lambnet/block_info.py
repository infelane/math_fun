# #todo things like the summary!
#
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import os, sys
import keras.backend as K
#
folder_loc = '/ipi/private/lameeus/private_Documents/python/2017_04'
cmd_subfolder = os.path.realpath(folder_loc)
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
#
import block_data
# import generate_h_images


class Info():
    def __init__(self, model):
        self.model = model
        

#     def set_train_data(self, train_data):
#         self._data_tr = train_data
#
#     def set_test_data(self, test_data):
#         self._data_te = test_data
#
#     # todo write summaries
#     def foo_summary(self):
#         ...

    def acc_summ(self, truth, pred):
        def bar(a, b):
            return np.sum(np.logical_and(a, b))

        y_hat_back = np.equal(pred, True)
        y_hat_loss = np.logical_not(y_hat_back)
        y_back = np.equal(truth, True)
        y_loss = np.logical_not(y_back)

        TP = bar(y_hat_loss, y_loss)
        TN = bar(y_hat_back, y_back)
        FN = bar(y_hat_back, y_loss)
        FP = bar(y_hat_loss, y_back)

        print("TP = {}".format(TP))
        print("FP = {}".format(FP))
        print("FN = {}".format(FN))
        print("TN = {}".format(TN))

        print("prec: {}".format(TN / (TN + FN)))
        print("sens: {}".format(TN / (TN + FP)))
    
    def output_vis(self, width, ext):
        set = 'zach'
        set = 'hand'
    
        data_input, map = block_data.test_data(set, width, ext, bool_new_data=False)
        
        n_depth = len(self.model.layers)
        
        outs = []
        
        for layer_i in range(n_depth):
            
            output_i = self.model.layers[layer_i].output
            
            shape_i = np.shape(output_i)
            ext_i = int((shape_i[1].value - width)/2.0)
        
            outs.append(output_i[...,ext_i:width+ext_i, ext_i:width+ext_i, :])
            
        func = K.function([self.model.layers[0].input], outs)
        
        generated_im = gen_image(func, data_input)

        n_depth = np.shape(generated_im)[-1]

        gen_im = []

        for i_im in range(n_depth):
            gen_im.append(generated_im[..., i_im])

        win_h = [400, 200]
        win_w = [200, 400]

        save_folder = '/media/lameeus/TOSHIBA/'

        from scipy.misc import imsave

        plt.figure()
        for i_im in range(n_depth):
            plt.subplot(5, 5, i_im+1)
    
            gen_im_i = gen_im[i_im]

            # mean = np.mean(gen_im_i)
            # std = np.std(gen_im_i)
            # extension = 3
            # vmin = mean - extension*std
            # vmax = mean + extension*std
            # vmin = np.min(gen_im_i)
            # vmax= np.max(gen_im_i)
            # plt.imshow(gen_im_i, vmin=vmin, vmax = vmax, cmap='seismic')
            
            plt.imshow(gen_im_i, cmap='seismic')

            # plt.axis([win_w[0], win_w[1], win_h[0], win_h[1]])
            
            plt.title('layer {}'.format(i_im + 1))

            imsave(save_folder + 'grey_{}.png'.format(i_im), gen_im_i)
            
            
        
        plt.show()
        plt.title('layer 10b')

    def output_test(self, width, ext, set = None):
        if not set:
            set = 'zach'
            # set = 'hand'

        data_input, map = block_data.test_data(set, width, ext, bool_new_data = False)

        generated_im = net2h_image(self, data_input, tophat_bool=False)

        gen_pred1 = generated_im[..., 1]

        out_pred1 = (np.greater(gen_pred1, 0.5)).astype(float)
        out_truth1 = data_input.im_out[..., 1]

        diff = np.zeros(shape=list(np.shape(out_truth1)) + [3])

        red = [1.0, 0.0, 0.0]
        green = [0.0, 1.0, 0.0]
        blue = [0.0, 0.0, 1.0]

        diff[np.logical_and(out_pred1 == 1.0, out_truth1 == 1.0)] =  red
        diff[np.greater(out_pred1, out_truth1)] = green
        diff[np.greater(out_truth1, out_pred1)] = blue

        map[out_pred1 == 1.0] = red

        self.acc_summ(out_truth1, out_pred1)

        folder = '/ipi/private/lameeus/data/lamb/output/'
        scipy.misc.imsave(folder + set + '_gen.png', diff[..., 1])
        scipy.misc.imsave(folder + set + '_gen_and_in.png', map)

        plt.figure()
        plt.subplot('321')
        plt.imshow(out_pred1, vmin=0.0, vmax=1.0, cmap='seismic')
        plt.title('pred loss')
        plt.subplot('322')
        plt.imshow(out_truth1, vmin=0.0, vmax=1.0, cmap='seismic')
        plt.title('truth loss')
        plt.subplot('323')
        plt.imshow(gen_pred1, vmin=0.0, vmax=1.0, cmap='seismic')
        plt.title('pred loss')
        ax = plt.subplot('324')
        plt.imshow(diff, vmin=0.0, vmax=1.0, cmap='seismic')
        plt.title('differenece')

        plt.subplot('325')
        plt.imshow(map)
        plt.title('map')

        plt.show()


def gen_image(func, data_input):
    in_patches = data_input.in_patches()
    batch_size = 100
    batch_amount = int(np.ceil(np.shape(in_patches)[0] / batch_size))
    
    out = []
    for batch_i in range(batch_amount):
        x = data_input.in_patches()[batch_i * batch_size:(batch_i + 1) * batch_size]

        out_i = np.concatenate(func([x]), axis = -1)
        out.append(out_i)
    out = np.concatenate(out, axis=0)
    im_lam = data_input.patches2images(out, normalize = False)

    # RIGHT
    x = data_input.right_patches().x
    out = np.concatenate(func([x]), axis = -1)
    im_right = data_input.right_patches2images(out, normalize = False)
    width = data_input.width
    shape_im = data_input.shape
    for h_i in range(int(shape_im[0] / width)):
        im_lam[h_i * width: (h_i + 1) * width, -width:, :] = im_right[h_i]
        
    # BOT
    x = data_input.bot_patches().x
    out = np.concatenate(func([x]), axis = -1)
    im_bot = data_input.bot_patches2images(out, normalize = False)
    for w_i in range(int(shape_im[1] / width)):
        im_lam[-width:, w_i * width: (w_i + 1) * width, :] = im_bot[w_i]
        
    # BOTRIGHT
    x = data_input.botright_patch().x
    out = np.concatenate(func([x]), axis = -1)
    im_bot_right = data_input.botright_patch2image(out, normalize = False)
    im_lam[-width:, -width:, :] = im_bot_right[...]
    
    return im_lam
    

def net2h_image(info=None, data=None, tophat_bool=True):

    layer_out = info.model.layers[-1].output[..., 0:2]

    func = K.function([info.model.layers[0].input], [layer_out])

    im_lam = gen_image(func, data)
    
    return im_lam