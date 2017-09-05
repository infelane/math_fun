# #todo things like the summary!
#

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

from f2017_04 import block_data
# import generate_h_images

from link_to_soliton.paint_tools import image_tools


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
    
    def output_vis(self, width, ext, set = 'zach', bool_save = False, last_layer = False):
        """
        :param width:
        :param ext:
        :param set: 'zach' or 'hand'
        :return:
        """

        # data_input, map = block_data.test_data(set, width, ext, bool_new_data=False)
        import block_data2
        data_input = block_data2.test_data(set, width, ext)
        
        n_depth = self.model.get_depth()
        
        if last_layer:
            n_depth_start = n_depth-1
        else:
            n_depth_start = 0
        
        outs = []
        n_outs = []
        
        for layer_i in range(n_depth_start, n_depth):
            
            output_i = self.model.get_conv_output(layer_i)
            # output_i = self.model.layers[layer_i].output
            
            shape_i = np.shape(output_i)
            ext_i = int((shape_i[1].value - width)/2.0)
        
            outs.append(output_i[...,ext_i:width+ext_i, ext_i:width+ext_i, :])
            n_outs.append(shape_i[-1])
            
        func = K.function([self.model.layers[0].input, K.learning_phase()], outs)
        
        generated_im = gen_image(func, data_input)

        gen_im = []
        
        n_images = np.sum(n_outs)
        
        for i_im in range(n_images):
            gen_im.append(generated_im[..., i_im])

        folder = '/ipi/private/lameeus/data/lamb/output/layers/'
        cmap = plt.cm.seismic

        for i_depth in range(len(n_outs)):
            plt.figure()
            
            n_outs_i = int(n_outs[i_depth])
            subplot_w = int(np.ceil(np.sqrt(n_outs_i)))
            subplot_h = int(np.ceil(n_outs_i/subplot_w))
            
            
            for j in range(n_outs_i):
                # gen_im contains all outputs from all layersc
                gen_im_ij = gen_im.pop(0)
                
                plt.subplot(subplot_h, subplot_w, j+1)
                
                if i_depth == len(n_outs) - 1:
                    vmin = 0.
                    vmax = 1.
                else:
                    vmin = np.percentile(gen_im_ij, 5)
                    vmax = np.percentile(gen_im_ij, 95)
                    
                plt.imshow(gen_im_ij, vmin = vmin, vmax = vmax, cmap=cmap, interpolation = 'nearest')

   
                if bool_save:
                    norm = plt.Normalize(vmin=vmin, vmax=vmax)
                    im_cmap = cmap(norm(gen_im_ij))
                    save_name = folder + set + '_l{}_{}'.format(i_depth, j)
                    plt.imsave(save_name, im_cmap)
                
            plt.title('layer: {}'.format(n_depth_start + i_depth + 1))
        plt.show()

    def output_test(self, width, ext, set = None):
        if not set:
            set = 'zach'
            # set = 'hand'

        data_input, map = block_data.test_data(set, width, ext, bool_new_data = False)
        import block_data2
        data_input = block_data2.test_data(set, width, ext)

        generated_im = net2h_image(self, data_input, tophat_bool=False)

        gen_pred0 = generated_im[..., 0]
        gen_pred1 = generated_im[..., 1]

        def show_histo(array):
            intens, bins = np.histogram(np.reshape(array, (-1)), bins=256, range=[-1, 2])
    
            bins_center = (bins[0:-1] + bins[1:]) / 2.0
    
            plt.plot(bins_center, intens)
            plt.show()
            
        show_histo(gen_pred0)
        show_histo(gen_pred1)

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
        scipy.misc.imsave(folder + set + '_gen.png', out_pred1)
        scipy.misc.imsave(folder + set + '_gen_and_in.png', map)
        # scipy.misc.imsave(folder + set + '_pred.png', out_pred1, cmap = 'seismic')

        # saves colormap
        cmap = plt.cm.seismic
        norm = plt.Normalize(vmin=0, vmax=1.0)
        im = cmap(norm(gen_pred1))
        plt.imsave(folder + set + '_pred.png', im)

        plt.figure()
        plt.subplot('321')
        plt.imshow(out_pred1, vmin=0.0, vmax=1.0, cmap='seismic')
        plt.title('pred loss')
        plt.subplot('322')
        plt.imshow(out_truth1, vmin=0.0, vmax=1.0, cmap='seismic')
        plt.title('truth loss')
        plt.subplot(3, 2, 3)
        plt.imshow(gen_pred0, vmin=0.0, vmax=1.0, cmap='seismic')
        plt.title('pred loss 0')
        plt.subplot(3, 2, 4)
        plt.imshow(gen_pred1, vmin=0.0, vmax=1.0, cmap='seismic')
        plt.title('pred loss 1')
        ax = plt.subplot(3, 2, 5)
        plt.imshow(diff, vmin=0.0, vmax=1.0, cmap='seismic')
        plt.title('differenece')

        plt.subplot(3, 2, 6)
        plt.imshow(map)
        plt.title('map')

        plt.show()

    
    def certainty(self, width, ext, set = None):
        
        version_nr = 'no_clean'
        
        if not set:
            set = 'zach'
            # set = 'hand'

        folder_loc = '/ipi/private/lameeus/private_Documents/python/2017_06'
        cmd_subfolder = os.path.realpath(folder_loc)
        if cmd_subfolder not in sys.path:
            sys.path.insert(0, cmd_subfolder)
        #


        if set == 'zach':
            im_clean = image_tools.path2im('/home/lameeus/data/ghent_altar/input/13_clean.tif')
        elif set == 'hand':
            im_clean = image_tools.path2im('/home/lameeus/data/ghent_altar/input/19_clean_crop_scale.tif')
        elif set == 'zach_small':
            im_clean = image_tools.path2im('/scratch/lameeus/data/ghentaltarpiece/altarpiece_close_up/beard_updated/rgb_cleaned.tif')
        elif set == 'hand_small':
            im_clean = image_tools.path2im(
                '/scratch/lameeus/data/ghentaltarpiece/altarpiece_close_up/finger/hand_cleaned.tif')

        # data_input, map = block_data.test_data(set, width, ext, bool_new_data = False)
        import block_data2
        data_input = block_data2.test_data(set, width, ext)

        generated_im = net2h_image(self, data_input)
    
        gen_pred0 = generated_im[..., 0]
        gen_pred1 = generated_im[..., 1]

        cert0 = 1-np.abs(gen_pred0 - 1.0)
        cert1 = 1-np.abs(gen_pred1 - 1.0)

        # uncertain =  np.abs(gen_pred0 - 1.0) + np.abs(gen_pred0 - 0.0)
        
        #
        # def show_histo(array):
        #     intens, bins = np.histogram(np.reshape(array, (-1)), bins=256, range=[-1, 2])
        #
        #     bins_center = (bins[0:-1] + bins[1:]) / 2.0
        #
        #     plt.plot(bins_center, intens)
        #     plt.show()
        #
        # show_histo(gen_pred0)
        # show_histo(gen_pred1)
        #
        # out_pred1 = (np.greater(gen_pred1, 0.5)).astype(float)
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
        # scipy.misc.imsave(folder + set + '_gen.png', out_pred1)
        # scipy.misc.imsave(folder + set + '_gen_and_in.png', map)
        # # scipy.misc.imsave(folder + set + '_pred.png', out_pred1, cmap = 'seismic')
        #
        # # saves colormap
        # cmap = plt.cm.seismic
        # norm = plt.Normalize(vmin=0, vmax=1.0)
        # im = cmap(norm(gen_pred1))
        # plt.imsave(folder + set + '_pred.png', im)
        #


        orange = np.reshape([1., 165./255., 0], (1,1,3))
        red = np.reshape([1., 0., 0], (1,1,3))

        im_clean[np.greater_equal(gen_pred1, 0.5)] = orange
        im_clean[np.greater_equal(gen_pred1, 0.8)] = red

        path = '/home/lameeus/data/ghent_altar/output/{}_7in_vhand_v{}.tif'.format(set, version_nr)
        image_tools.save_im(im_clean, path)
        
        plt.imshow(im_clean)
        plt.show()
        
        plt.figure()
        # plt.subplot('321')
        # plt.imshow(out_pred1, vmin=0.0, vmax=1.0, cmap='seismic')
        # plt.title('pred loss')
        # plt.subplot('322')
        # plt.imshow(out_truth1, vmin=0.0, vmax=1.0, cmap='seismic')
        # plt.title('truth loss')
        plt.subplot(3, 2, 3)
        plt.imshow(gen_pred0, vmin=0.0, vmax=1.0, cmap='seismic')
        plt.title('pred loss 0')
        plt.subplot(3, 2, 4)
        plt.imshow(gen_pred1, vmin=0.0, vmax=1.0, cmap='seismic')
        plt.title('pred loss 1')
        plt.subplot(3, 2, 5)
        plt.imshow(cert0, vmin=0.0, vmax=1.0, cmap='seismic')
        plt.title('certainty of class 0')
        plt.subplot(3, 2, 6)
        plt.imshow(cert1, vmin=0.0, vmax=1.0, cmap='seismic')
        plt.title('certainty of class 1')
        

        
        # plt.subplot(3, 2, 6)
        # plt.imshow(map)
        # plt.title('map')
        #
        plt.show()
        
    
    def certainty_discr(self, width, ext, set = None):
        
        if set == 'zach':
            im_clean = image_tools.path2im('/home/lameeus/data/ghent_altar/input/13_clean.tif')
        elif set == 'hand':
            im_clean = image_tools.path2im('/home/lameeus/data/ghent_altar/input/19_clean_crop_scale.tif')
        elif set == 'zach_small':
            im_clean = image_tools.path2im(
                '/scratch/lameeus/data/ghentaltarpiece/altarpiece_close_up/beard_updated/rgb_cleaned.tif')
        elif set == 'hand_small':
            im_clean = image_tools.path2im(
                '/scratch/lameeus/data/ghentaltarpiece/altarpiece_close_up/finger/hand_cleaned.tif')
        
        folder_loc = '/ipi/private/lameeus/private_Documents/python/2017_06'
        cmd_subfolder = os.path.realpath(folder_loc)
        if cmd_subfolder not in sys.path:
            sys.path.insert(0, cmd_subfolder)
        import block_data2
        from f2017_06 import block_data2
        
        data_input = block_data2.test_data(set, width, ext)

        generated_im = net2h_image_discr(self, data_input)

        gen_pred0 = generated_im[..., 3]  # background
        gen_pred1 = generated_im[..., 4]  # paint loss
        gen_pred2 = generated_im[..., 5]  # segmented
        
        
        
        # gen_pred1 = generated_im[..., 1]

        # cert0 = 1 - np.abs(gen_pred0 - 1.0)
        # cert1 = 1 - np.abs(gen_pred1 - 1.0)

        # orange = np.reshape([1., 165. / 255., 0], (1, 1, 3))
        # red = np.reshape([1., 0., 0], (1, 1, 3))

        # im_clean[np.greater_equal(gen_pred1, 0.5)] = orange
        # im_clean[np.greater_equal(gen_pred1, 0.8)] = red

        plt.figure()
        plt.subplot(3, 2, 1)
        plt.imshow(im_clean)
        plt.title('original input')
        plt.subplot(3, 2, 3)
        plt.imshow(gen_pred0, vmin=0.0, vmax=1.0, cmap='seismic')
        plt.title('discr: background')
        plt.subplot(3, 2, 4)
        plt.imshow(gen_pred1, vmin=0.0, vmax=1.0, cmap='seismic')
        plt.title('discr: paint loss')
        plt.subplot(3, 2, 5)
        plt.imshow(gen_pred2, vmin=0.0, vmax=1.0, cmap='seismic')
        plt.title('discr: segmented with computer')
        
  

        plt.show()

def gen_image(func, data_input):
    in_patches = data_input.in_patches()
    batch_size = 100
    batch_amount = int(np.ceil(np.shape(in_patches)[0] / batch_size))
    
    out = []
    for batch_i in range(batch_amount):
        x = data_input.in_patches()[batch_i * batch_size:(batch_i + 1) * batch_size]

        out_i = np.concatenate(func([x, 0]), axis = -1)
        out.append(out_i)
    out = np.concatenate(out, axis=0)
    im_lam = data_input.patches2images(out, normalize = False)

    # RIGHT
    x = data_input.right_patches().x
    out = np.concatenate(func([x, 0]), axis = -1)
    im_right = data_input.right_patches2images(out, normalize = False)
    width = data_input.width
    shape_im = data_input.shape
    for h_i in range(int(shape_im[0] / width)):
        im_lam[h_i * width: (h_i + 1) * width, -width:, :] = im_right[h_i]
        
    # BOT
    x = data_input.bot_patches().x
    out = np.concatenate(func([x, 0]), axis = -1)
    im_bot = data_input.bot_patches2images(out, normalize = False)
    for w_i in range(int(shape_im[1] / width)):
        im_lam[-width:, w_i * width: (w_i + 1) * width, :] = im_bot[w_i]
        
    # BOTRIGHT
    x = data_input.botright_patch().x
    out = np.concatenate(func([x, 0]), axis = -1)
    im_bot_right = data_input.botright_patch2image(out, normalize = False)
    im_lam[-width:, -width:, :] = im_bot_right[...]
    
    return im_lam


def gen_image_discr(func, data_input):
    # TODO

    in_patches = data_input.in_patches()
    out_patches = data_input.out_patches()
    batch_size = 100
    batch_amount = int(np.ceil(np.shape(in_patches)[0] / batch_size))

    out = []
    for batch_i in range(batch_amount):
        x = in_patches[batch_i * batch_size:(batch_i + 1) * batch_size]
        y = out_patches[batch_i * batch_size:(batch_i + 1) * batch_size]
    
        # out_i = np.concatenate((func([x, y]))[0], axis=-1)
        # out_i =(func([x, y]))[0]
        out_i = np.concatenate(func([x, y]), axis = 3)

        # print(np.shape((func([x, y]))[0]))
        # print(np.shape(out_i))
        
        out.append(out_i)
    out = np.concatenate(out, axis=0)
    im_lam = data_input.patches2images(out, normalize=False)
    
    return im_lam

def net2h_image(info=None, data=None):  # , tophat_bool=True
    # TODO, is this even okey?
    # if isinstance(info.model.output, list):
    #
    # else:
    layer_out = info.model.output[..., :]
    # layer_out = info.model.output[0][..., :]
    
    func = K.function([info.model.input, K.learning_phase()], [layer_out])
    im_lam = gen_image(func, data)
    return im_lam


def net2h_image_discr(info, data):
    # layer_out = info.model.output[..., :]
    # func = K.function([info.model.input, K.learning_phase()], [layer_out])
    func = info.model.predict_auto
    im_lam = gen_image_discr(func, data)
    return im_lam
