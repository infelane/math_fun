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
    
    model = keras_ipi.block_builder.stack(layers)

    optimizer = {'class_name': 'adam', 'config': {'lr': flag.lr}} #otherwise  = 'adam'
    
    # loss = 'categorical_crossentropy' # TODO find out if I can change this: loss = {'class_name': 'categorical_crossentropy', 'config' : {}}
    # keras.losses.categorical_crossentropy
    
    loss = keras_ipi.losses.weigthed_crossentropy(layers.w_c)
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
        model.load_weights(filepath, depth = 2)
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
            model.fit(X_train[:100000, ...], Y_train[:100000, ...],
                      batch_size=64, epochs=1, verbose=1, shuffle=True,
                      # class_weight= (1.0, 10.0),
                      validation_data=(X_test, Y_test), callbacks=callbacks_list)


    #     model.save_weights(filepath)
    # model.save_weights(filepath)
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
    
    # info.output_vis(8, 7)

    # keras_ipi.
    
    keras_ipi.results.roc(model, X_test, Y_test)
    

if __name__ == '__main__':
    main()
    