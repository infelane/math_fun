#todo things like the summary!

import matplotlib.pyplot as plt
import numpy as np
import scipy

import block_data
import generate_h_images


class Info():
    def __init__(self, network_group):
        self.network_group = network_group
        
    def set_train_data(self, train_data):
        self._data_tr = train_data
        
    def set_test_data(self, test_data):
        self._data_te = test_data
        
    # todo write summaries
    def foo_summary(self):
        ...
        
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
        
                
    def output_test(self, width, ext):
        set = 'zach'
        # set = 'hand'

        data_input, map = block_data.test_data(set, width, ext, bool_new_data = True)

        # data_input
        
        generated_im = generate_h_images.net2h_image(self.network_group, data_input, tophat_bool=False)

        gen_pred1 = generated_im[..., 1]
        
        out_pred1 = (np.greater(gen_pred1, 0.3)).astype(float)
        out_truth1 = data_input.im_out[..., 1]

        # diff = np.copy(out_pred0)

        diff = np.zeros(shape=list(np.shape(out_truth1)) + [3])
        
        red = [1.0, 0.0, 0.0]
        green = [0.0, 1.0, 0.0]
        blue = [0.0, 0.0, 1.0]
        
        diff[np.logical_and(out_pred1 == 1.0, out_truth1 == 1.0)] =  red
        diff[np.greater(out_pred1, out_truth1)] = green
        diff[np.greater(out_truth1, out_pred1)] = blue

        map[out_pred1 == 1.0] = red

        self.acc_summ(out_truth1, out_pred1)
        
        scipy.misc.imsave('/ipi/private/lameeus/data/tensorflow/output/gen.png', diff[..., 1])
        scipy.misc.imsave('/ipi/private/lameeus/data/tensorflow/output/gen_and_in.png', map)

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
        