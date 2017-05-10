# should be cleaner than main2
# probs will convert everything to keras

import keras
import lambnet
from keras import backend as K

import os, sys
import numpy as np

#3th party
folder_loc = '/ipi/private/lameeus/private_Documents/python/2017_January/tensorflow_folder'
cmd_subfolder = os.path.realpath(folder_loc)
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
# import config_lamb
import data

import pickle
import config3


load_prev = True
def main():
    if load_prev:
        batch_train = pickle.load(open("batch_train.p", "rb"))
        batch_test = pickle.load(open("batch_test.p", "rb"))
        
    else:
        width = 8
        data_all = data.ground_truth(width=width, ext=7)
        
        batch_train = data_all[0].next_batch(100000)
        batch_test = data_all[1].get_test_data()
    
        pickle.dump(batch_train, open("batch_train.p", "wb"))
        pickle.dump(batch_test, open("batch_test.p", "wb"))

    # todo
    X_train = batch_train.x
    Y_train = batch_train.y
    X_test = batch_test.x
    Y_test = batch_test.y
    
    flag = config3.flag()
    layers = config3.nn()
    
    model = lambnet.block_builder.stack(layers)
    
    optimizer = {'class_name': 'adam', 'config': {'lr': flag.lr}} #otherwise  = 'adam'
    # loss = 'categorical_crossentropy' # TODO find out if I can change this: loss = {'class_name': 'categorical_crossentropy', 'config' : {}}
    # loss = [loss, loss]
    
    import functools

    # todo
    def w_categorical_crossentropy(y_true, y_pred, weights = np.ones((2,))):
        
        # nb_cl = len(weights)
        
        # from itertools import product
        # final_mask = K.zeros_like(y_pred[..., 0])
        # y_pred_max = K.max(y_pred, axis=1)
        # y_pred_max = K.expand_dims(y_pred_max, 1)
        # y_pred_max_mat = K.equal(y_pred, y_pred_max)
        # for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        #     final_mask += (
        #     K.cast(weights[c_t, c_p], K.floatx()) * K.cast(y_pred_max_mat[:, c_p], K.floatx()) * K.cast(y_true[..., c_t],
        #                                                                                                 K.floatx()))
        # return K.categorical_crossentropy(y_pred, y_true) * final_mask
        import tensorflow as tf
        a = tf.multiply(y_true * tf.log(y_pred), weights) # tf.cast(weights, dtype=tf.float32))
        cost = -tf.reduce_mean(a)
        return cost
    
    loss = lambda a, b : w_categorical_crossentropy(a, b, weights=np.asarray(layers.w_c))
    # loss = functools.partial(w_categorical_crossentropy, weights = ) # to add aditional parameters
    # loss = w_categorical_crossentropy
    
    # def metric_foo(y_true, y_pred):
    #
    #
    #
    #     return keras.metrics.categorical_accuracy(y_true, y_pred)
        
    
    # metric_i = lambnet.metrics.accuracy_test
    TP = lambnet.metrics.TP
    FP = lambnet.metrics.FP
    FN = lambnet.metrics.FN
    TN = lambnet.metrics.TN
    sens = lambnet.metrics.sens
    prec = lambnet.metrics.prec

    # todo
    model.compile(loss = loss,
                  optimizer=optimizer,
                  metrics=['accuracy', sens, prec]
                  )
    
    filepath = 'foo_weight.h5'
    
    file_pre = '/scratch/Downloads_local/vgg16_weights.h5'


    if flag.bool_prev:
        model.load_weights(filepath, layer_i = [0,3])
        # model.load_weights(file_pre)

    checkpoint = keras.callbacks.ModelCheckpoint(filepath, verbose=0,
                                    save_weights_only=True, period=1)

    callbacks_list = [checkpoint]
    
    #TODO
    if True:
        epochs = 1
        for i in range(epochs):
            print('epoch {}/{}'.format(i, epochs))
            model.fit(X_train, Y_train,
                      batch_size=64, epochs=1, verbose=1, shuffle=True,
                      validation_data=(X_test, Y_test), callbacks=callbacks_list)

            # model.save_weights('foo_weight.h5')

            # model.save_weights('foo_weight.h5', layer_i=[0, 3])
            # model.save_weights('foo_weight_0.h5', layer_i = [0,2])

    #     model.save_weights(filepath)
    # model.save_weights(filepath)

    # # lambnet.
    #
    # # print(model.summary())
    #
    #
    # get_last_layer_output = K.function([model.layers[0].input], [model.layers[-1].output])
    #
    # print(get_last_layer_output([X_test])[0])
    # #
    # score = model.evaluate(X_test, Y_test, batch_size=100, verbose=0)
    # print(score)
    # #
    info = lambnet.block_info.Info(model)

    # info.output_test(8, 7, set='hand')
    # info.output_test(8, 7, set='zach')
    
    info.output_vis(8, 7)
    

if __name__ == '__main__':
    main()
    