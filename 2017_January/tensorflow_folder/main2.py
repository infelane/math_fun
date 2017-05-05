# Main file for Lamb-net

# 3th party libraries
import os
import pickle
import sys

import config_lamb
import data
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

folder_loc = '/home/lameeus/Documents/Link to Python/2017_February/super_res_challenge'
cmd_subfolder = os.path.realpath(folder_loc)
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
folder_loc = '/home/lameeus/Documents/Link to Python'
cmd_subfolder = os.path.realpath(folder_loc)
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import lambnet #import network
import net_builder
import block_info


def default_save(saver, sess, flag, global_epoch):
    """ In order to make sure all saves are similar"""
    print('-- saving --')
    saver.save(sess, flag.checkpoint_dir + 'chpt', global_step=global_epoch, write_meta_graph= False)

def train_loop(network_group: lambnet.network.NetworkGroup, funcs, data_all, flag, info: block_info.Info):
    sess = network_group.sess
    training_data = data_all[0]
    validation_data = data_all[1]

    keep_prob = network_group.keep_prob
    
    am_mini_batch = int(np.ceil(training_data.num_examples / flag.batch_size))

    # todo
    am_mini_batch = int(am_mini_batch/10)

    train_data = training_data.get_test_data()
    valid_data = validation_data.get_test_data()
    info.set_train_data(train_data)
    info.set_test_data(valid_data)

    for step in range(flag.training_epochs):
        for _ in range(am_mini_batch): # Go over all the mini batches

            batch = training_data.next_batch(flag.batch_size)
            
            feed_dict = {network_group.x: batch.x, network_group.y: batch.y, keep_prob: flag.dropout }

            network_group.train(feed_dict)
                    
        epoch = network_group.plus_global_epoch()
  
        cost = network_group.cost(feed_dict = {network_group.x:valid_data.x,
                                               network_group.y:valid_data.y})
        print("Validation cost: {}".format(cost))

        if epoch % flag.checkpoint_steps_size == 0:
            network_group.save_params(flag.checkpoint_dir)

            feed_dict = {network_group.x: valid_data.x, network_group.y: valid_data.y}
            valAcc, conf, sens, prec = sess.run([funcs.accuracy, funcs.conf, funcs.sensitivity, funcs.precision], feed_dict=feed_dict)
            valCost = network_group.cost(feed_dict)

            feed_dict = {network_group.x: train_data.x, network_group.y: train_data.y}
            summary = sess.run(network_group.merged, feed_dict=feed_dict)
            
            feed_dict = {network_group.x: valid_data.x, network_group.y: valid_data.y}
            summary_test = sess.run(network_group.merged_test, feed_dict=feed_dict)
            
            network_group.update_summary(summary)
            network_group.update_summary_test(summary_test)

            print("vali accuracy %g" % valAcc)
            print("vali cost {}".format(valCost))
            # print("vali confusion {}".format(conf))
            print("vali sensitivity {}".format(sens))
            print("vali precision {}".format(prec))
            
            # summary_op = conf_plot_summ(conf, epoch)
            # summary = sess.run(summary_op)
            # network_group.update_summary(summary)
            
            # summary = sess.run(network_group.summ1, feed_dict=feed_dict)
            #
            # network_group.update_summary_conf(summary)

            # print("train weighted cost %g" % np.mean(valWeightedCost))

    # After last training step, make sure it is saved

    network_group.save_params(flag.checkpoint_dir)
    print('-- learning done --')
    
all_conf = [[], [], [], []]
time = []
    
def conf_plot_summ(conf, epoch):
    """Create a pyplot plot and save to buffer."""
    all_conf[0].append(conf[0, 0])
    all_conf[1].append(conf[0, 1])
    all_conf[2].append(conf[1, 0])
    all_conf[3].append(conf[1, 1])
    time.append(epoch)
       
    plt.figure()
    plt.plot(time, all_conf[0], label='TP')
    plt.plot(time, all_conf[1], label='FN')
    plt.plot(time, all_conf[2], label='FP')
    plt.plot(time, all_conf[3], label='TN')
    plt.title("Confusion matrix")
    plt.legend
    
    # Prepare the plot
    plot_buf = gen_tb_plot()
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    # Add image summary
    return tf.summary.image("plot", image, collections=['all'])


import io

def gen_tb_plot():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.clf()
    return buf
    

# TODO use summary
class Funcs():
    """ Saves all the important function for training, plotting ... """
    def __init__(self, placeholders, flag, global_epoch):
        """ The training step """
        y_guess = placeholders.ff
        
        self.foo = tf.argmax(y_guess, -1)
        
        correct_prediction = tf.equal(tf.argmax(y_guess, -1), tf.argmax(placeholders.y, -1))

        y_hat_softmax = tf.nn.softmax(placeholders.ff)
        self.cost = -tf.reduce_sum(placeholders.y * tf.log(y_hat_softmax), [1])
        # self.cost = tf.nn.softmax_cross_entropy_with_logits(placeholder.ff, placeholder.y)
        cross_entropy = tf.reduce_mean(self.cost, name='xentropy_mean')
        self.train = tf.train.AdamOptimizer(flag.lr).minimize(cross_entropy)
        #TODO self.train = tf.train.GradientDescentOptimizer(flag.lr).minimize(cross_entropy)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.increment_epoch = tf.assign(global_epoch, global_epoch + 1)
        
        true_loss = tf.greater(placeholders.y[..., 1], 0.5)
        true_back = tf.greater_equal(placeholders.y[..., 0], 0.5)
        pred_loss = tf.greater(y_guess[..., 1], 0.5)
        pred_back = tf.greater_equal(y_guess[..., 0], 0.5)
        
        def my_operator(a, b):
            return tf.reduce_sum(tf.cast(tf.logical_and(a, b), tf.int32))
        
        TP_val = my_operator(true_loss, pred_loss)
        FN_val = my_operator(true_loss, pred_back)
        FP_val = my_operator(true_back, pred_loss)
        TN_val = my_operator(true_back, pred_back)

        # confusion matrix
        self.conf = tf.stack([[TP_val, FN_val], [FP_val, TN_val]])
        
        # Probability of detecting: percentage of loss that is correctly identified
        true = tf.reduce_sum(self.conf, axis=1)
        self.sensitivity = tf.divide(tf.diag_part(self.conf), true)
        # # probability of a label being correct
        pred = tf.reduce_sum(self.conf, axis=0)
        self.precision = tf.divide(tf.diag_part(self.conf), pred)
        
        with tf.name_scope('confusion'):
            tf.summary.scalar('TP', self.conf[0, 0], collections=['all'])
            tf.summary.scalar('FN', self.conf[0, 1], collections=['all'])
            tf.summary.scalar('TN', self.conf[1, 1], collections=['all'])
            
        with tf.name_scope('accuracy'):
            tf.summary.scalar('sens_loss', self.sensitivity[0], collections=['all'])
            tf.summary.scalar('sens_back', self.sensitivity[1], collections=['all'])
            tf.summary.scalar('prec_loss', self.precision[0], collections=['all'])
            tf.summary.scalar('prec_back', self.precision[1], collections=['all'])
                                           
    def set_cost_train(self, placeholder, flag):
        """ add a weight to the cost """
        #TODO clean up
        k_def = 2
        tensor_k = tf.convert_to_tensor(k_def, dtype=tf.float32)
        k = tf.placeholder_with_default(tensor_k, tf.TensorShape(()))
        k_weights = [(k + 1) / (2 * k), (k + 1) /2]
        k_weights2 = tf.Variable(k_weights, name='cost_weight', trainable=False)
        cost_rescale = tf.gather(k_weights2, tf.argmax(placeholder.y, 1))
        cost_LAM = tf.multiply(self.cost, cost_rescale, name='weighted_cost')
        cross_entropy_LAM = tf.reduce_mean(cost_LAM, name='weighted_xentropy')
        self.train = tf.train.AdamOptimizer(flag.lr).minimize(cross_entropy_LAM)
        # self.train = tf.train.GradientDescentOptimizer(flag.lr).minimize(cross_entropy_LAM)


def main():
    ### Settings ###
    
    net_base = net_builder.base_lamb()
    network_group = net_base.network_group
    
    info = block_info.Info(network_group)
    
    flag = config_lamb.FLAGS1()
    
    # # inits
    # layers = config_lamb.nn4()
    #
    # network_group = network.NetworkGroup(layers=layers, bool_residue = False)
    
    x = network_group.x
    y = network_group.y
    
    network_group.set_lr(lr = flag.lr, mom = flag.mom)

    # TODO
    # # learnable clipping val
    # clipping_val = tf.Variable(1.0e10, name='clipping', trainable= True)
    # network_group.params.update({clipping_val.name: clipping_val})
    # tf.summary.scalar('clipping', clipping_val, collections = ['all'])
    
    bool_clipping = True
    network_group.set_train(bool_clipping=bool_clipping)


    ### Data ###
    # data_all = data.data_ex1(new=False)
    # data_all = data.data_ex2(new=False, ext_in = 8, ext_out = 7, n_i = 100)

    width = network_group.a_layers.layer_size[-1][0]
        

    ### Training functions ###
    funcs = Funcs(network_group, flag, network_group.global_epoch)
    funcs.set_cost_train(network_group, flag)

    sess = network_group.sess

    init = tf.global_variables_initializer()
    sess.run(init)

    # The summary
    network_group.set_summary(flag.summary_dir)
    network_group.set_summary_conf(flag.summary_dir, funcs.conf)
    
    if flag.load_prev:  # Restore previous
        network_group.load_params(flag.checkpoint_dir)
    else:
        network_group.load_init()

    #TODO pruning
    network_group.pruning()

    if flag.train:      # Train the network
        data_all = data.ground_truth(width=width, ext=7)
    
        train_loop(network_group, funcs, data_all, flag, info)
        
    # foo(network_group, network_group, width = width)
    info.output_test(width, ext = 7)

    # # gaussian_numbers = np.random.randn(1000)
    # for key, value in params.items():
    #     foo = value.eval()
    #     print(foo)
    #     size = np.size(foo)
    #     if size > 1:
    #         w_f, bin_border = np.histogram(foo)
    #         bins = (bin_border[:-1] + bin_border[1:])/2
    #
    #         print(bins)
    #         plt.plot(bins, w_f/size, alpha=1, label = key)
    #
    # # plt.title("Gaussian Histogram")
    # # plt.xlabel("Value")
    # plt.ylabel("Frequency")
    # plt.legend(loc='upper right')
    # # fig = plt.gcf()
    # plt.show()

    data_te = data_all[2]
    feed_dict = {x: data_te.images, y: data_te.labels}

    bins_center, distr, cumsum = labels_cumm(sess, funcs, feed_dict, network_group)

    plot_labels_cumm(bins_center, distr, cumsum)

    plot_data = PlotData()
    plot_data.set_cumsum(bins_center, distr, cumsum)

    data_dict = {'bins_center': bins_center, 'distr': distr, 'cumsum': cumsum}
    pickle.dump(data_dict, open('/scratch/lameeus/data/tensorflow/results/plot_data.p', "wb"))
    
    A = np.array([bins_center, distr[0], cumsum[0]], dtype=float)
    B = np.array([bins_center, distr[1], cumsum[1]], dtype=float)
    np.savetxt('/scratch/lameeus/data/tensorflow/results/plot_data_0.csv', np.transpose(A), delimiter=',', header='bins_center, distr, cumsum')
    np.savetxt('/scratch/lameeus/data/tensorflow/results/plot_data_1.csv', np.transpose(B), delimiter=',', header='bins_center, distr, cumsum')

    # feed_dict = {x: data_tr.images, y: data_tr.labels}
    # plot_(sess, feed_dict, funcs, placeholders)
    
    # feed_dict = {x: data_va.images, y: data_va.labels}
    # plot_(sess, feed_dict, funcs, placeholders)
 
    return network_group, plot_data


class PlotData(object):
    def __init__(self):
        ...

    def set_cumsum(self, bins_center, distr, cumsum):
        self.bins_center = bins_center
        self.distr = distr
        self.cumsum = cumsum
        


def plot_labels_cumm(bins_center, distr, cumsum):
    plt.figure(1)
    plt.subplot(121)
    plt.plot(bins_center[::-1], distr[0], 'b+--', label='label 0')
    plt.plot(bins_center, distr[1], 'r+--', label='label 1')
    plt.xlabel('Guess of label')
    legend = plt.legend(loc='upper center')
    plt.title("Histogram")

    plt.subplot(122)
    plt.plot(bins_center[::-1], cumsum[0], 'b+--', label='label 0')
    plt.plot(bins_center, cumsum[1], 'r+--', label='label 1')
    plt.xlabel('guess label')
    legend = plt.legend(loc='upper center')
    plt.title("Cummulative distribution")
    plt.show()

# generates cummulative distribution of labels
def labels_cumm(sess =  None, funcs = None, feed_dict = None, network_group = None):
    
    # todo split up in smaller bits
    acc, sens, prec  = sess.run([funcs.accuracy, funcs.sensitivity,
                                       funcs.precision], feed_dict=feed_dict)

    cost = network_group.cost(feed_dict = feed_dict)

    print("test accuracy {}".format(acc))
    print("test cost {}".format(cost))
    print("test sensitivity {}".format(sens))
    print("test precision {}".format(prec))

    predictions = sess.run([network_group.ff, network_group.y], feed_dict=feed_dict)

    pred = predictions[0]
    truth = predictions[1]

    distr_star = [[], []]
    histo_mat = np.zeros(shape = (2, 2), dtype=int)

    pred_flat = np.reshape(pred, newshape=(-1, 2))
    truth_flat = np.reshape(truth, newshape=(-1, 2))

    pred_args = np.argmax(pred_flat, axis=-1).astype(np.int8)
    truth_args = np.argmax(truth_flat, axis=-1).astype(np.int8)
       
    for index, truth_i in enumerate(truth_args):
        
        pred_args_i = pred_args[index]
        pred_i = pred_flat[index, truth_i]
        
        histo_mat[pred_args_i, truth_i] += 1
        distr_star[truth_i].append(pred_i)
        
    bins = np.arange(0, 1.01, 0.01)
          
    distr0, _ = np.histogram(distr_star[0], bins=bins)
    distr1, _ = np.histogram(distr_star[1], bins=bins)

    distr = np.array([distr0, distr1])

    bins_center = np.mean([bins[:-1], bins[1:]], axis=0)
    cumsum = np.cumsum(distr[..., ::-1], axis=1)[..., ::-1]

    return(bins_center, distr, cumsum)

if __name__ == '__main__':
    main()
