# For data analysis

import tensorflow as tf
import numpy as np

#### __init__.py
import config_lamb
####


def main():
    config1 = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    config1.gpu_options.per_process_gpu_memory_fraction = 0.9
    
    flag = config_lamb.FLAGS1()

    sess = tf.Session(config=config1)

    ckpt = tf.train.get_checkpoint_state(flag.checkpoint_dir)
    print(ckpt.model_checkpoint_path)
    # tf.train.Saver()
    
    # saver = tf.train.Saver() # params
    
    
    print(tf.contrib.framework.list_variables(flag.checkpoint_dir))
    a = tf.contrib.framework.load_checkpoint(ckpt.model_checkpoint_path)
    print(a)

    saved_shapes = a.get_variable_to_shape_map()
    print(saved_shapes)

    params_d = {}
    
    for b in saved_shapes:
        print(b)
        c = tf.Variable(tf.zeros(shape = saved_shapes[b]))
        
        params_d.update({b: c})

    saver = tf.train.Saver(params_d)
    
    path = ckpt.model_checkpoint_path
    saver.restore(sess, path)
    
    e = params_d[b]
    f = sess.run(e)
    
    print(f)
    print(np.shape(f))
    
    # tf.tra

if __name__ == '__main__':
    main()