# #todo things like the summary!
#
import matplotlib.pyplot as plt
import numpy as np
import scipy
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
        
        # layer_10 = model.layers[-1].output[..., 0:2]
        # layer_9 = model.layers[-2].output[..., 0:2]
        # layer_8 = model.layers[-3].output[..., 0:2]
        # # see what it learned at previous layer
        # self.get_last_layer_output = K.function([model.layers[0].input], [layer_8, layer_9, layer_10])

#     def set_train_data(self, train_data):
#         self._data_tr = train_data
#
#     def set_test_data(self, test_data):
#         self._data_te = test_data
#
#     # todo write summaries
#     def foo_summary(self):
#         ...
#
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
        
        
        layer_10 = self.model.layers[-1].output[..., 0:1]
        layer_9 = self.model.layers[-2].output[..., 0:1]
        layer_8 = self.model.layers[-3].output[..., 0:1]

        func = K.function([self.model.layers[0].input], [layer_8, layer_9, layer_10])
        
        
        # func = K.function([self.model.layers[0].input], [self.model.layers[-2].output[..., 0:2]])
        
        generated_im = gen_image(func, data_input)
        
        gen_im1 = generated_im[..., 0]
        gen_im2 = generated_im[..., 1]
        gen_im3 = generated_im[..., 2]

        # gen_pred1 = generated_im[..., 1]
        #
        # out_pred1 = (np.greater(gen_pred1, 0.3)).astype(float)
        # out_truth1 = data_input.im_out[..., 1]
        #
        # diff = np.zeros(shape=list(np.shape(out_truth1)) + [3])
        #
        # red = [1.0, 0.0, 0.0]
        # green = [0.0, 1.0, 0.0]
        # blue = [0.0, 0.0, 1.0]
        #
        # diff[np.logical_and(out_pred1 == 1.0, out_truth1 == 1.0)] = red
        # diff[np.greater(out_pred1, out_truth1)] = green
        # diff[np.greater(out_truth1, out_pred1)] = blue
        #
        # map[out_pred1 == 1.0] = red
        #
        # self.acc_summ(out_truth1, out_pred1)
        #
        # folder = '/ipi/private/lameeus/data/lamb/output/'
        # scipy.misc.imsave(folder + set + '_gen.png', diff[..., 1])
        # scipy.misc.imsave(folder + set + '_gen_and_in.png', map)
        #
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(gen_im1, cmap='seismic')
        plt.title('8')

        plt.subplot(2, 2, 2)
        plt.imshow(gen_im2, cmap='seismic')
        plt.title('9')

        plt.subplot(2, 2, 3)
        plt.imshow(gen_im3, cmap='seismic')
        plt.title('10')
        
        # plt.title('pred loss')
        # plt.subplot('322')
        # plt.imshow(out_truth1, vmin=0.0, vmax=1.0, cmap='seismic')
        # plt.title('truth loss')
        # plt.subplot('323')
        # plt.imshow(gen_pred1, vmin=0.0, vmax=1.0, cmap='seismic')
        # plt.title('pred loss')
        # ax = plt.subplot('324')
        # plt.imshow(diff, vmin=0.0, vmax=1.0, cmap='seismic')
        # plt.title('differenece')
        #
        # plt.subplot('325')
        # plt.imshow(map)
        # plt.title('map')
        #
        plt.show()

        
    
    def output_test(self, width, ext):
        set = 'zach'
        set = 'hand'

        data_input, map = block_data.test_data(set, width, ext, bool_new_data = True)

        generated_im = net2h_image(self, data_input, tophat_bool=False)

        gen_pred1 = generated_im[..., 1]

        out_pred1 = (np.greater(gen_pred1, 0.3)).astype(float)
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
    # out = []
    
    out = None


    for batch_i in range(batch_amount):
        x = data_input.in_patches()[batch_i * batch_size:(batch_i + 1) * batch_size]
        # feed_dict = {info.x: data.in_patches()[batch_i * batch_size:(batch_i + 1) * batch_size]}
        # if tophat_bool:
        #     feed_dict.update(
        #         {info.x_tophat: data.in_patches_gausshat()[batch_i * batch_size:(batch_i + 1) * batch_size]})
        #
        # out_i = info.get_output(feed_dict=feed_dict)
    
        # out_i = func([x])[0]

        out_i = np.concatenate(func([x]), axis = -1) # func([x])[0]
        
    #     out.append(out_i)
    #
    # out = np.concatenate(out, axis=0)
    #
    # im_lam = data_input.patches2images(out)
    #
    # return im_lam

        if out is None:
            out = out_i
        else:
            out = np.append(out, out_i, 0)

    im_lam = data_input.patches2images(out)
    
    return im_lam
    

def net2h_image(info=None, data=None, tophat_bool=True):
    # Split up the output in smaller patches
    in_patches = data.in_patches()
    batch_size = 100
    batch_amount = int(np.ceil(np.shape(in_patches)[0] / batch_size))
    out = None
    
    for batch_i in range(batch_amount):
        x = data.in_patches()[batch_i * batch_size:(batch_i + 1) * batch_size]
        # feed_dict = {info.x: data.in_patches()[batch_i * batch_size:(batch_i + 1) * batch_size]}
        # if tophat_bool:
        #     feed_dict.update(
        #         {info.x_tophat: data.in_patches_gausshat()[batch_i * batch_size:(batch_i + 1) * batch_size]})
        #
        # out_i = info.get_output(feed_dict=feed_dict)
        
        out_i = info.get_last_layer_output([x])[0]
        
        if out is None:
            out = out_i
        else:
            out = np.append(out, out_i, 0)
    
    im_lam = data.patches2images(out)
    
    # def build_dict(data_placeholder):
    #     feed_dict = {info.x: data_placeholder.x}
    #     if tophat_bool:
    #         feed_dict.update({info.x_tophat: data_placeholder.x_tophat})
    #     return feed_dict
    #
    # data_placeholder = data.right_patches()
    # feed_dict = build_dict(data_placeholder)
    # out = info.get_output(feed_dict=feed_dict)
    #
    # im_right = data.right_patches2images(out)
    #
    # width = data.width
    #
    # data_placeholder = data.botright_patch()
    # feed_dict = build_dict(data_placeholder)
    # out = info.get_output(feed_dict=feed_dict)
    # im_lam[-width:, -width:, :] = data.botright_patch2image(out)
    #
    # shape_im = data.shape
    # for h_i in range(int(shape_im[0] / width)):
    #     im_lam[h_i * width: (h_i + 1) * width, -width:, :] = im_right[h_i]
    #
    # data_placeholder = data.bot_patches()
    # feed_dict = build_dict(data_placeholder)
    # out = info.get_output(feed_dict=feed_dict)
    #
    # im_bot = data.bot_patches2images(out)
    #
    # shape_im = data.shape
    # for w_i in range(int(shape_im[1] / width)):
    #     im_lam[-width:, w_i * width: (w_i + 1) * width, :] = im_bot[w_i]
    #
    # # plt.imshow(foo[0 : 1344, :, :])
    # # plt.show()
    #
    # # shape = np.shape(im_inp_float)
    # #
    # # h = shape[0]
    # # w = shape[1]
    # #
    # # im_lam = np.zeros(shape=(2 * h, 2 * w, 3))
    # #
    # # for h_i in range(h):
    # #     inp_i = np.zeros((w, 3, 3, 3))
    # #     for w_i in range(w):
    # #         inp_i[w_i, ...] = image_extended.get_segm(h_i, w_i)
    # #
    # #     feed_dict = {placeholders.x: inp_i}
    # #     out_i = networkGroup.get_output(placeholders, feed_dict=feed_dict)
    # #
    # #     for w_i in range(w):
    # #         im_lam[2 * h_i:2 * h_i + 2, 2 * w_i:2 * w_i + 2, :] = out_i[w_i, ...]
    # #
    # # # TODO in_image => placeholders.x: input, get output, output to output_im
    # #
    # # im_lam[im_lam > 1.0] = 1.0
    # # im_lam[im_lam < 0.0] = 0.0
    
    return im_lam