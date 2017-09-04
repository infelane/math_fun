import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

device_name = 'gpu'  # Choose device from cmd line. Options: gpu or cpu
size = 100
shape = (int(size), int(size))
if device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"

with tf.device(device_name):
    random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)


startTime = datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        result = session.run(sum_operation)
        print(result)

# It can be hard to see the results on the terminal with lots of output -- add some newlines to improve readability.
print("\n" * 5)
print("Shape:", shape, "Device:", device_name)
print("Time taken:", datetime.now() - startTime)

print("\n" * 5)

#
# def part2():
#     sess = tf.Session()
#     new_saver = tf.train.import_meta_graph('my-model.meta')
#     new_saver.clean_init(sess, tf.train.latest_checkpoint('./'))
#     all_vars = tf.get_collection('vars')
#     for v in all_vars:
#         v_ = sess.run(v)
#         print('{} -> {}'.format(v.name, v_))
#
# # part1()
# # part2()
#
# def part3():
#     import tensorflow as tf
#
#     import os
#     dir = os.path.dirname(os.path.realpath(__file__))
#
#     # First, you design your mathematical operations
#     # We are the default graph scope
#
#     # Let's design a variable
#     v1 = tf.Variable(1., name="v1")
#     v2 = tf.Variable(2., name="v2")
#     # Let's design an operation
#     a = tf.add(v1, v2)
#
#     # Let's create a Saver object
#     # By default, the Saver handles every Variables related to the default graph
#     all_saver = tf.train.Saver()
#     # But you can precise which vars you want to save under which name
#     v2_saver = tf.train.Saver({"v2": v2})
#
#     # By default the Session handles the default graph and all its included variables
#     with tf.Session() as sess:
#         # Init v and v2
#         sess.run(tf.global_variables_initializer())
#         # Now v1 holds the value 1.0 and v2 holds the value 2.0
#         # We can now save all those values
#
#         all_saver.save(sess, dir + '/results/data-all.chkp')
#         # or saves only v2
#         v2_saver.save(sess, dir + '/results/data-v2.chkp')
#
# # part3()
#
# def part4():
#     import tensorflow as tf
#
#     # Let's load a previously saved meta graph in the default graph
#     # This function returns a Saver
#     saver = tf.train.import_meta_graph('results/data-all.chkp.meta')
#
#     # We can now access the default graph where all our metadata has been loaded
#     graph = tf.get_default_graph()
#
#     # Finally we can retrieve tensors, operations, collections, etc.
#     global_step_tensor = graph.get_tensor_by_name('loss/global_step:0')
#     train_op = graph.get_operation_by_name('loss/train_op')
#     hyperparameters = tf.get_collection('hyperparameters')
#
#     with tf.Session() as sess:
#         # To initialize values with saved data
#         saver.clean_init(sess, 'results/model.ckpt.data-1000-00000-of-00001')
#         print(sess.run(global_step_tensor))  # returns 1000
#
#
#
# part4()
#