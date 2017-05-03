import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

# Own packages
folder_loc = '/home/lameeus/Documents/Link to Python/2017_January/tensorflow_folder'
cmd_subfolder = os.path.realpath(folder_loc)
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import generate_h_images
from lambnet import network
import data_net
import config
import data_net_sr


# Input of a neural network
class InputImage():
    def __init__(self, low_im):
        # output of first method (input of the network)
        im_bicubic = scipy.misc.imresize(low_im, 2.0, interp='bicubic') / 255
        self.x = im_bicubic


def main():
    ### Load Settings
    layers = config.layer_config3()
    flag = config.FLAGS1()
        
    ### Build
    patches_input0 = data_net_sr.gen_patches_input_shuffled(bool_new=False)
    patches_input1 = data_net_sr.gen_patches_input_shuffled(bool_new=False, part = 1)

    data_test = data_net_sr.data_test()
    network_group = network.NetworkGroup(layers=layers, bool_residue=True)  # Don't set before funcs?
    
    ### Set settings
    network_group.set_lr(lr = flag.lr)
    network_group.set_gradient_clipping(clip_val = flag.clip_val)
    
    ### load data
        
    #TODO cleanup

    # for filename in glob.glob(refdir + '/*.png'): #assuming gif
    #     im=Image.open(filename)
    #     image_list.append(np.array(im))
    #     print('{}\t:{}'.format(index, filename))
    #     index+=1 #Update the index





    # inputImage = InputImage(im_inp_float)
    
    # save_folder = '/scratch/lameeus/data/challenges/NTIRE17/'
    #
    # output_file = 'test.png'
    #
    # scipy.misc.toimage(im_resized, cmin=0.0, cmax=1.0).save(
    #     save_folder + output_file)

    # show2figures(im_ref, im_inp)

    # width = layers.layer_size[0][0]

    

    



    
    # print(np.shape(data_train.in_patches()))
    
 



    # ONLY RED:
    # foo_input = foo_input[...,0]
    # foo_output = foo_output[...,0]

    # performance(im_resized, im_inp)


    # network_group.set_train(lr = flag.lr)

    # x = network_group.x
    # y = network_group.y
    # ff = network_group.ff

    # placeholders = Placeholders(ff, x, y)
    
    network_group.set_summary(flag.summary_dir)

    if flag.depth_train:
        network_group.set_train_dep(flag.lr, depth = flag.depth_train)
   
    ### Training functions ###
    # funcs = Funcs(placeholders, flag, global_epoch)

    network_group.load_variables()

    # ### Initializations ###
    #
    # sess = network_group.sess
    # # with tf.device("/gpu:0"):

    if flag.load_prev:  # Restore previous
        # ckpt = tf.train.get_checkpoint_state(flag.checkpoint_dir)
        # print(ckpt.model_checkpoint_path)
        # # Overwrites the necessairy values by restoring variables from disk.
        #
        # # saver_load.restore(sess, ckpt.model_checkpoint_path)
        network_group.load_params(flag.checkpoint_dir)

    if flag.train:  # Train the network
        ### Data ###
        data_tr = data_net.DataSet2(images=patches_input0[0], images_that=patches_input0[1], labels=patches_input0[2])
        # data_tr1 = data_net.DataSet2(images=patches_input_shuffled1, images_that =  labels=patches_output_shuffled1)
        data_tr1 = data_net.DataSet2(images=patches_input1[0], images_that=patches_input1[1], labels=patches_input1[2])
    
        train_loop(network_group, (data_tr, data_tr1), flag)
        # train_loop(network_group, (None, data_tr1), flag, global_epoch=global_epoch)

    show_results(network_group, data_test= data_test)

def show_results(networkGroup = None,
                 data_test = None):
    
    im_lam = generate_h_images.net2h_image(networkGroup = networkGroup,
                                           data = data_test
                                           )

    networkGroup.close_session()

    (_, snr_cnn) = performance(im_lam, data_test.goal_im())
    (_, snr_bicubic) = performance(data_test.ref_im(), data_test.goal_im())

    x_min = 1050
    x_max = 1150
    y_min = 750
    y_max = 700
    
    def plot_foo(diff):
        plt.imshow(diff, vmin=-0.5, vmax=0.5, interpolation="nearest", cmap ="seismic")
        plt.colorbar()
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

    im_i = data_test.ref_im()
    plt.subplot(321)
    plt.imshow(im_i, vmin=0, vmax=1, interpolation="nearest")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title('Bicubic, psnr = {}'.format(snr_bicubic))
    plt.subplot(322)
    diff = np.mean(data_test.goal_im() - im_i, axis=-1)
    plot_foo(diff)

    plt.subplot(323)
    plt.imshow(im_lam, vmin=0, vmax=1, interpolation="nearest")
    plt.title('LAM, psnr = {}'.format(snr_cnn))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.subplot(324)
    diff = np.mean(data_test.goal_im() - im_lam, axis=-1)
    plot_foo(diff)

    plt.subplot(325)
    plt.imshow(data_test.goal_im(), vmin=0, vmax=1, interpolation="nearest")
    plt.title('GOAL')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.subplot(326)
    plt.imshow(data_test.goal_im() - data_test.goal_im(), vmin=0, vmax=1)
    plt.show()


class Placeholders():
    def __init__(self, ff, x, y):
        self.ff = ff
        self.x = x
        self.y = y

    def add_w(self, w):
        self.w = w



def default_save(saver, sess, flag, global_epoch):
    print('-- saving --')
    saver.save(sess, flag.checkpoint_dir + 'chpt', global_step=global_epoch, write_meta_graph=False)


def train_loop(network_group, training_data, flag):
    am_mini_batch = int(np.ceil(training_data[1].num_examples / flag.batch_size))
    am_mini_batch = 10   #100
    
    sess = network_group.sess

    for step in range(flag.training_epochs):
        for _ in range(am_mini_batch):  # Go over all the mini batches

            # batch0 = training_data[0].next_batch(flag.batch_size)
            batch1 = training_data[1].next_batch(flag.batch_size)
            
            # x = np.concatenate([batch0.x, batch1.x], axis=0)
            # # y = np.concatenate([batch0[1], batch1[1]], axis=0)
            # y = np.concatenate([batch0.y, batch1.y], axis=0)
            # x_tophat = np.concatenate([batch0.x_tophat, batch1.x_tophat], axis=0)

            x = batch1.x
            x_tophat = batch1.x_tophat
            y = batch1.y

            # feed_dict = {network_group.x: x, network_group.y: y, network_group.x_tophat = x_tophat}
            feed_dict = {network_group.x: x,
                         network_group.x_tophat:x_tophat,
                         network_group.y:y}

            if flag.depth_train:
                network_group.train_depth(feed_dict, depth=flag.depth_train)

            else:
                network_group.train(feed_dict)
            
        # training_data[0].batch_zero()
        # training_data[1].batch_zero()
        # training_data[0].shuffle()
        training_data[1].shuffle()
        
        ### for the shearlet thing
        test_batch = training_data[1].get_test_data()
        feed_dict = {network_group.x: test_batch.x,
                     network_group.x_tophat: test_batch.x_tophat,
                     network_group.y: test_batch.y}
        # cost = sess.run(funcs.cost_mean, feed_dict=feed_dict)
        cost = network_group.cost(feed_dict)
        # cost_depth = sess.run(network_group.cost_depth, feed_dict=feed_dict)

        print('shearlet {}'.format(cost))


        # test_batch = training_data[0].get_test_data()
        # feed_dict = {network_group.x: test_batch.x,
        #              network_group.x_tophat: test_batch.x_tophat,
        #              network_group.y: test_batch.y}
        # # cost = sess.run(funcs.cost_mean, feed_dict=feed_dict)
        # cost = network_group.cost(feed_dict)
        # # cost_depth = sess.run(network_group.cost_depth, feed_dict=feed_dict)
        #
        # print('bicubic {}'.format(cost))
        
        if flag.depth_train:
            cost_depth = network_group.cost_depth(feed_dict, depth = flag.depth_train)
            print("cost of depth: {}".format(cost_depth))

        epoch = network_group.plus_global_epoch()
        
        if epoch % flag.checkpoint_steps_size == 0:
            # default_save(saver, sess, flag, global_epoch)
            network_group.save_params(flag.checkpoint_dir)
            
            summary = sess.run(network_group.merged, feed_dict = feed_dict)
            network_group.update_summary(summary)

            # test_batch = training_data[0].get_test_data()
            # feed_dict = {network_group.x: test_batch.x,
            #              network_group.x_tophat: test_batch.x_tophat,
            #              network_group.y: test_batch.y}
            # network_group.update_summary_i('data1', feed_dict)
            test_batch = training_data[1].get_test_data()
            feed_dict = {network_group.x: test_batch.x,
                         network_group.x_tophat: test_batch.x_tophat,
                         network_group.y: test_batch.y}
            network_group.update_summary_i('data2', feed_dict)


def weight_variable(shape, index):
    stddev = 0.01
    initial = tf.truncated_normal(shape, stddev=stddev, name="init_W_" + str(index))

    var = tf.Variable(initial, name='W_' + str(index))
    return var
                

# Abstract class
from abc import ABCMeta


class Layer(metaclass=ABCMeta):
    @staticmethod
    # @abstractmethod
    def my_abstract_staticmethod(self, x):
        ...

    def set_input(self):
        ...

    def get_output(self):
        ...

    def new_func(self):
        ...
    
    
def show2figures(im1, im2):
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(im1)
    plt.subplot(122)
    plt.imshow(im2)
    plt.show()


def performance(im1, im2, ssim_bool = True):
    if ssim_bool:
        a1 = ssim(im1, im2, multichannel=True)
    else:
        a1 = 0
    a2 = psnr(im1, im2)

    print("1-ssim = {0} (should be close to 1) \npsnr = {1} (as high as possible)\n".format(1-a1, a2))

    return(a1, a2)


if __name__ == '__main__':
    main()
