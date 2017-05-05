# import tensorflow as tf
import numpy as np

# Extension of the Saver class of TF
class Save_network(object):
	def __init__(self, tf):
		#TODO
		print("TODO")

		self.saver = tf.train.Saver(tf.all_variables())
		self.tf = tf

	def save(self, sess, name = 'my-model'):

		self.saver.save(sess, name)

	def load(self, name):
		sess = self.tf.Session()
		new_saver = self.tf.train.import_meta_graph(name + '.meta')
		new_saver.clean_init(sess, self.tf.train.latest_checkpoint('./'))
		all_vars = self.tf.get_collection('vars')

		# Now each var is gone over twice
		for v in all_vars:
			v_ = sess.run(v)
			print('{}: {}'.format(v.name, np.shape(v_)))

		return sess