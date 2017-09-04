# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

### Data ###
import data

import numpy as np

(data_tr, data_va, data_te) = data.data_ex1(new=False)

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 63])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

W = tf.Variable(tf.zeros([63,2]))
b = tf.Variable(tf.zeros([2]))

sess.run(tf.global_variables_initializer())

y = tf.matmul(x,W) + b

def reshaper(foo):
    return np.reshape(foo, newshape=(-1, 63))


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(1000):
  batch = data_tr.next_batch(100)
  train_step.run(feed_dict={x: reshaper(batch[0]), y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: reshaper(data_te.images), y_: data_te.labels}))

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')

W_conv1 = weight_variable([1, 1, 7, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,3,3,7])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)
h_pool1 = h_conv1

W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)
h_pool2 = h_conv2

W_fc1 = weight_variable([1 * 1 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 1*1*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cost = tf.nn.softmax_cross_entropy_with_logits(y_conv, y_)
cross_entropy = tf.reduce_mean(cost)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

k_def = 30
tensor_k = tf.convert_to_tensor(k_def, dtype=tf.float32)
k = tf.placeholder_with_default(tensor_k, tf.TensorShape(()))
k_weights = [(k + 1) / (2 * k), (k + 1) /2]
k_weights2 = tf.Variable(k_weights, trainable=False)
cost_rescale = tf.gather(k_weights2, tf.argmax(y_, 1))
cost_LAM = tf.mul(cost, cost_rescale)
cross_entropy_LAM = tf.reduce_mean(cost_LAM)
# train_step = tf.train.GradientDescentOptimizer(1e-8).minimize(cross_entropy_LAM)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
tr_cost = 0
for i in range(20000):
  batch = data_tr.next_batch(50)
  if i%100 == 0:
    feed_dict = {x:reshaper(batch[0]), y_: batch[1], keep_prob: 1.0}
    train_accuracy = accuracy.eval(feed_dict=feed_dict)
    tr_cost = cross_entropy_LAM.eval(feed_dict=feed_dict)
    print("step %d, training accuracy %g, tr cost = %g"%(i, train_accuracy, tr_cost))

    # foo = cost_LAM.eval(feed_dict=feed_dict)
    # _sum_and = cost.eval(feed_dict=feed_dict)
    # print(foo)
    # print(sum_and)

  train_step.run(feed_dict={x: reshaper(batch[0]), y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: reshaper(data_te.images), y_: data_te.labels, keep_prob: 1.0}))
