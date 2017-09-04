import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def example3():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    sess = tf.InteractiveSession()


    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def conv_pool(x, W, b):
        tf.add_to_collection('vars', W)
        tf.add_to_collection('vars', b)
        z = tf.nn.relu(conv2d(x, W) + b)
        return max_pool_2x2(z)

    def fully_con_layer(x, W, b):
        tf.add_to_collection('vars', W)
        tf.add_to_collection('vars', b)
        #TODO get shape of W
        flat_x = tf.reshape(h_pool2, [-1, widht_layer * widht_layer * 64])
        return tf.nn.relu(tf.matmul(flat_x, W) + b)

    x = tf.placeholder(tf.float32, shape=[None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # h_pool1 = max_pool_2x2(h_conv1)
    h_pool1 = conv_pool(x_image, W_conv1, b_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # h_pool2 = max_pool_2x2(h_conv2)
    h_pool2 = conv_pool(h_pool1, W_conv2, b_conv2)

    # Should be 4, not 7!?
    #TODO why is this 7 and not 4?? Other kind of convolution?
    widht_layer = 7
    W_fc1 = weight_variable([widht_layer * widht_layer * 64, 1024])
    b_fc1 = bias_variable([1024])

    # h_pool2_flat = tf.reshape(h_pool2, [-1, widht_layer * widht_layer * 64])
    # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1 = fully_con_layer(h_pool2, W_fc1, b_fc1)

    # For dropout, placeholder in order to being able to switch of or on
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.initialize_all_variables())
    # prev range 20000 (acc = 99.2)
    for i in range(100):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    # Save the session

    from save_network import Save_network
    saver = Save_network(tf)

    saver.save(sess, 'my-model')

    sess.run(tf.initialize_all_variables())

    sess = saver.load('my-model')

    for i in range(100):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

import sys, os
# cmd_subfolder = os.path.realpath(
# 	os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "subfolder")))
# sys.path.append( <path to dirFoo> )
folder_loc = '/home/lameeus/Documents/Link to Python/2016_November/PhD/packages'
cmd_subfolder = os.path.realpath(folder_loc)
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
from training_data import Training_data

from tf_network import *

def example4():

    # n = (1000, 1000, 1000)
    # images_set = 'beard'
    # ext_out = 0
    # training_data = Training_data(new = False, amount = 3000, ext_out = ext_out, images_set = images_set)
    # training_data_gpu, validation_data_gpu, test_data_gpu = training_data.split_theano(n = n)

    # print(np.shape(training_data_gpu[0].eval()))
    # print(np.shape(training_data_gpu[1].eval()))

    # layer_size = [[21, 21], [17, 17], [13, 13], [9, 9], [5, 5], [1, 1]]
    # kernel_size = [[5,5], [5,5], [5,5], [5,5], [5,5]]
    # kernel_depth = [7, 10, 10, 10, 10, 2]
    # layers = Layers(layer_size, kernel_size, kernel_depth)

    """ Build the layers """
    layer_size = [[21, 21], [17, 17], [1, 1]]
    kernel_size = [[5,5], [17, 17]]
    kernel_depth = [7, 5, 2]
    layers = Layers(layer_size, kernel_size, kernel_depth)

    # layers.save('test_layer')
    # layers.load('test_layer')

    for v in tf.global_variables():
        print('{} -> '.format(v.name))

    #TODO use the "Get_collection"

    """ Network """
    ff = layers.get_ff()
    x = layers.get_x()  #Placeholder
    y = layers.get_y()  #Placeholder
    y_guess = tf.nn.softmax(ff)

    """ Data """
    import data
    (data_tr, data_va, data_te) = data.data_ex1(False)

    def CreateSaver():
        saver = tf.train.Saver()
        return saver

    """ Saver """
    saver_params = tf.train.Saver()

    foo = tf.get_collection_ref('params')
    for v in foo:
        print('?: {}'.format(v.name))

    class FLAGS1():
        train = True
        training_steps = 100
        checkpoint_steps = 10
        checkpoint_dir = '/scratch/data/tensorflow/'
        summary_dir = checkpoint_dir + 'summary/'
        load_prev = True
        lr = 1e-4

    flag = FLAGS1()

    # restore last session
    def restore_session(saver):
        ckpt = tf.train.get_checkpoint_state(flag.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.clean_init(sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError("Can't find checkpoints")

        print("variables restored:")
        all_vars = tf.global_variables()
        for v in all_vars:
            print('{} -> '.format(v.name))
        print("\n")

        sess.run(init)

    with tf.Session() as sess:

        """ default values """
        k_def = 30
        keep_prob_def = 1.0

        """ Variables """
        tensor_keep_prob = tf.convert_to_tensor(keep_prob_def, dtype=tf.float32)
        keep_prob = tf.placeholder_with_default(tensor_keep_prob, tf.TensorShape(()))

        tensor_k = tf.convert_to_tensor(k_def, dtype=tf.float32)
        k = tf.placeholder_with_default(tensor_k, tf.TensorShape(()))
        k_weights = [(k + 1) / (2 * k), (k + 1) /2]
        k_weights2 = tf.Variable(k_weights, name='cost_weight', trainable=False)

        global_step = tf.Variable(0, name='global_step', trainable=False)

        """ Functions """
        # Standard cost
        def include_cost():
            cost = tf.nn.softmax_cross_entropy_with_logits(ff, y)
            cross_entropy = tf.reduce_mean(cost, name='xentropy_mean')
            train_step = tf.train.AdamOptimizer(flag.lr).minimize(cross_entropy, global_step=global_step)
            # Add to summary
            tf.summary.scalar(cross_entropy.op.name, cross_entropy)
            return train_step, cost

        # Lam-Cost
        def include_lam_cost(cost):
            cost_rescale = tf.gather(k_weights2, tf.argmax(y, 1))
            cost_LAM = tf.mul(cost, cost_rescale, name='weighted_cost')
            cross_entropy_LAM = tf.reduce_mean(cost_LAM, name='weighted_xentropy')
            train_step = tf.train.AdamOptimizer(flag.lr).minimize(cross_entropy_LAM, global_step=global_step)
            # Add to summary
            tf.summary.scalar(cross_entropy_LAM.op.name, cross_entropy_LAM)
            return train_step, cost_LAM

        train1, cost1 = include_cost()
        train2, cost2 = include_lam_cost(cost1)

        train_step = train1

        # Other
        correct_prediction = tf.equal(tf.argmax(y_guess, 1), tf.argmax(y, 1))
        # reduce mean gives same results as np.mean
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        """ Initialization """
        init = tf.global_variables_initializer()

        """ Network loading """
        #Load the previous
        if flag.load_prev:
            print("Restore previous")
            saver = CreateSaver()
            restore_session(saver)

        else:
            print("Initialize with new values")
            sess.run(init)

        # print('? {}'.format(sess.run(global_step)))

        #TODO works till here

        # try to find the latest checkpoint in my_checkpoint_dir, then create a session with that restored
        # if no such checkpoint, then call the init_op after creating a new session
        # sess = sm.prepare_session("", init_op=init, saver=saver, checkpoint_dir=flag.checkpoint_dir)

        # foo = tf.train.global_step(sess, global_step)
        # print('restored global step: {}'.format(foo))

        def training():

            # Collecting all summaries (at this point only the xentropy
            summary = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(flag.summary_dir, sess.graph)

            foo = tf.get_collection_ref('params')
            for v in foo:
                print('?: {}'.format(v.name))

            # with tf.Graph().as_default():
            class Ugh:
                def __enter__(self):
                    print('a')
                    return 'a'

                def __exit__(self, type, value, traceback):
                    print('a')

            with Ugh() as print_:

                foo = tf.get_collection_ref('params')
                for v in foo:
                    print('?: {}'.format(v.name))

                # sess.run(init)

                for i in range(flag.training_steps):
                    batch = data_tr.next_batch(200)
                    # sess.run(train_step, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
                    """ The weighted cost function """
                    feed_dict = {x: batch[0], y: batch[1], keep_prob: 0.5}
                    sess.run(train_step, feed_dict= feed_dict)
                    # train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

                    foo = tf.train.global_step(sess, global_step)
                    print(foo)

                    # Update the summaries
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, i)

                    """ checkpoints """
                    if (i + 1) % flag.checkpoint_steps == 0:
                        saver.save(sess, flag.checkpoint_dir + 'model.ckpt',
                                   global_step=global_step)

                        feed_dict = {x: data_te.images, y: data_te.labels, keep_prob: 1.0}
                        valAcc, valCost, valWeightedCost = sess.run([accuracy, cost1, cost2], feed_dict=feed_dict)

                        print_prog()

                        print("test accuracy %g" % valAcc)
                        print("train cost %g" % np.mean(valCost))
                        print("train weighted cost %g" % np.mean(valWeightedCost))

                        # Update the summaries
                        summary_str = sess.run(summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, i)

                        """ Testingg!? """
                        # all_vars = tf.get_collection_ref('params')
                        # print("Can I find some params?")
                        # for v in all_vars:
                        #     v_ = sess.run(v)
                        #     print('{} -> {}'.format(v.name, v_))

                        foo = tf.get_collection_ref('params')
                        for v in foo:
                            print('?: {}'.format(v.name))

        def print_prog():
            feed_dict = {x: data_te.images, y: data_te.labels, keep_prob: 1.0}
            valAcc, valCost, valWeightedCost = sess.run([accuracy, cost1, cost2], feed_dict=feed_dict)

            print("test accuracy %g" % valAcc)
            print("train cost %g" % np.mean(valCost))
            print("train weighted cost %g" % np.mean(valWeightedCost))


        if flag.train:
            """ training loop """
            saver = CreateSaver()
            training()

        else:
            saver = CreateSaver()
            restore_session(saver)

            print_prog()

            # Now you can run the model to get predictions

            predictions = sess.run([y_guess, y], feed_dict={x: data_tr.images, y: data_tr.labels})
            print(predictions[0])
            print(predictions[1])

            foo = predictions[0]
            bar = predictions[1]

            histo = []
            for i in range(2):
                histo_i = []
                for j in range(1000):
                    if np.argmax(bar[j,:] == i):
                        histo_i.append(foo[j,i])
                # histo_i = [y_guess_i[i] for y_guess_i in y_guess if argmax() ]
                histo.append(np.array(histo_i))

            bins = np.arange(0,1.01,0.01)
            distr0 = np.histogram(histo[0], bins=bins)
            distr1 = np.histogram(histo[1], bins=bins)

            # bins = a[1]
            distr = np.array([distr0[0], distr1[0]])
            bins_center = np.mean([bins[:-1], bins[1:]], axis = 0)
            cumsum = np.cumsum(distr, axis = 1)

            plt.figure(1)
            plt.subplot(121)
            plt.plot(bins_center[::-1], distr[0], 'b+--', label = 'label 0')
            plt.plot(bins_center, distr[1], 'r+--', label = 'label 1')
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




    # sess = tf.InteractiveSession()
    #
    # sess.run(tf.initialize_all_variables())
    #
    # W_b = sess.run([ff, y], feed_dict= {x: data_tr.images, y: data_tr.labels})
    # print(np.shape(W_b[0]), np.shape(W_b[1]))
    #
    # # prev range 20000 (acc = 99.2)
    # for i in range(1000):
    #     batch = data_tr.next_batch(200)
    #     if i % 100 == 0:
    #         train_accuracy = accuracy.eval(feed_dict={
    #             x: batch[0], y: batch[1], keep_prob: 1.0})
    #         print("step %d, training accuracy %g" % (i, train_accuracy))
    #     train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
    #


if True:
    # example2()
    # example3()
    example4()

