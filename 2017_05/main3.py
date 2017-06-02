# should be cleaner than main2
# probs will convert everything to keras

import os
import sys

import keras

# some_file.py
# sys.path.insert(0, '/ipi/private/lameeus/private_Documents')
# from ipi.private.lameeus.private_Documents import keras_ipi
import keras_ipi
import lambnet


import config3

#3th party
folder_loc = '/ipi/private/lameeus/private_Documents/python/2017_January/tensorflow_folder'
cmd_subfolder = os.path.realpath(folder_loc)
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
# import config_lamb
import data

import pickle


load_prev = True #TODO set to True
def main():
    if load_prev:
        batch_train = pickle.load(open("batch_train.p", "rb"))
        batch_vali = pickle.load(open("batch_vali.p", "rb"))
        batch_test = pickle.load(open("batch_test.p", "rb"))
        
    else:
        width = 8
        data_all = data.ground_truth(width=width, ext=7)
        
        batch_train = data_all[0].next_batch(100000)
        batch_vali = data_all[1].next_batch(10000)
        batch_test = data_all[2].next_batch(10000)
    
        pickle.dump(batch_train, open("batch_train.p", "wb"))
        pickle.dump(batch_vali, open("batch_vali.p", "wb"))
        pickle.dump(batch_test, open("batch_test.p", "wb"))

    # todo
    # subset
    n_subset = 100000 # 10000 for small subset, 100000 for all
    X_train = batch_train.x[:n_subset]
    Y_train = batch_train.y[:n_subset]
    X_vali = batch_vali.x[:n_subset]
    Y_vali = batch_vali.y[:n_subset]
    X_test = batch_test.x[:n_subset]
    Y_test = batch_test.y[:n_subset]
    
    flag = config3.flag()
    layers = config3.nn()
    
    model = keras_ipi.block_builder.stack(layers)
    
    print(model.summary())

    optimizer = {'class_name': 'adam', 'config': {'lr': flag.lr, 'beta_1': flag.beta}} #otherwise  = 'adam'
    
    # loss = 'categorical_crossentropy' # TODO find out if I can change this: loss = {'class_name': 'categorical_crossentropy', 'config' : {}}
    # keras.losses.categorical_crossentropy
    
    loss = keras_ipi.losses.weigthed_crossentropy(k = layers['k'], r = layers['r'])
    # loss = keras.losses.categorical_crossentropy
    # loss = keras.losses.mean_squared_error
    # loss = keras.losses.mean_absolute_error
    # loss = 'categorical_crossentropy'
    
    TP = keras_ipi.metrics.TP
    FP = keras_ipi.metrics.FP
    FN = keras_ipi.metrics.FN
    TN = keras_ipi.metrics.TN
    sens = keras_ipi.metrics.sens
    prec = keras_ipi.metrics.prec

    # todo
    model.compile(loss = loss,
                  optimizer=optimizer,
                  metrics=['accuracy', sens, prec]
                  )
    
    filepath = 'foo_weight.h5'
    
    # file_pre = '/scratch/Downloads_local/vgg16_weights.h5'


    if flag.bool_prev:
        depth = len(model.layers)
        model.load_weights(filepath, depth = depth)
        # model.load_weights(file_pre)

    checkpoint = keras.callbacks.ModelCheckpoint(filepath, verbose=0,
                                    save_weights_only=True, period=1)
    
    # summary = keras.callbacks.RemoteMonitor(root = 'http://localhost:9000',
    #                                         path = '/scratch/lameeus/data/lamb/summ_keras/',
    #                                         field='data',
    #                                         headers = None)

    summary = keras.callbacks.TensorBoard(log_dir='/scratch/lameeus/data/lamb/summ_keras/',
                                          histogram_freq=1,
                                          write_graph=False,
                                          write_images=False,
                                          embeddings_freq=0,
                                          embeddings_layer_names=None,
                                          embeddings_metadata=None)

    callbacks_list = [checkpoint, summary]
    
    #TODO
    if True:
        epochs = flag.epochs
        for i in range(epochs):
            print('epoch {}/{}'.format(i, epochs))
            model.fit(X_train, Y_train,
                      batch_size=flag.batch_size, epochs=1,  shuffle=True,
                      verbose=1,    # how much information to show 1 much or 0, nothing
                      # class_weight= (1.0, 10.0),
                      validation_data=(X_test[:10000], Y_test[:10000]), callbacks=callbacks_list)

            score = model.evaluate(X_train[:10000], Y_train[:10000], batch_size=flag.batch_size, verbose=0)
            print(score)
            score = model.evaluate(X_test[:10000], Y_test[:10000], batch_size=flag.batch_size, verbose=0)
            print(score)

            keras_ipi.results.roc(model, X_vali[:10000], Y_vali[:10000], auc_only=True) # hand
            keras_ipi.results.roc(model, X_test[:10000], Y_test[:10000], auc_only=True) # Zach

    #     model.save_weights(filepath)
    # model.save_weights(filepath)
    #
    # # print(model.summary())
    #
    # get_last_layer_output = K.function([model.layers[0].input], [model.layers[-1].output])
    #
    # print(get_last_layer_output([X_test])[0])
    # #
    # score = model.evaluate(X_test, Y_test, batch_size=100, verbose=0)
    # print(score)
    
    info = lambnet.block_info.Info(model)

    info.output_test(8, 7, set='hand')
    info.output_test(8, 7, set='zach')

    info.output_vis(8, 7)
    #
    # # keras_ipi.
    # import numpy as np
    # print(np.shape(X_train)[0])

    # keras_ipi.results.roc(model, X_train[11000:12000], Y_train[11000:12000], auc_only=False)
    # keras_ipi.results.roc(model, X_vali, Y_vali, auc_only = False)   # hand
    # keras_ipi.results.roc(model, X_test, Y_test, auc_only = False)   # Zach


if __name__ == '__main__':
    main()
    