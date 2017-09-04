# should be cleaner than main2
# probs will convert everything to keras

import pickle
import keras

# 3th party
from f2017_05 import config3
import keras_ipi
from f2017_01.tensorflow_folder import data

load_prev = True    # TODO set to True

def main():
    folder_data = '/home/lameeus/data/ghent_altar/input_arrays/main3/'
    if load_prev:
        batch_train = pickle.load(open(folder_data + "batch_train.p", "rb"))
        batch_vali = pickle.load(open(folder_data + "batch_vali.p", "rb"))
        batch_test = pickle.load(open(folder_data + "batch_test.p", "rb"))
        
    else:
        width = 8
        data_all = data.ground_truth(width=width, ext=7)
        
        batch_train = data_all[0].next_batch(100000)
        batch_vali = data_all[1].next_batch(10000)
        batch_test = data_all[2].next_batch(10000)
    
        pickle.dump(batch_train, open(folder_data + "batch_train.p", "wb"))
        pickle.dump(batch_vali, open(folder_data + "batch_vali.p", "wb"))
        pickle.dump(batch_test, open(folder_data + "batch_test.p", "wb"))

    # subset
    n_subset = 100000   # 10000 for small subset, 100000 for all
    
    x_train = batch_train.x[:n_subset]
    y_train = batch_train.y[:n_subset]
    x_vali = batch_vali.x[:n_subset]
    y_vali = batch_vali.y[:n_subset]
    x_test = batch_test.x[:n_subset]
    y_test = batch_test.y[:n_subset]

    datagen = keras_ipi.preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,  # 20
        width_shift_range=0.2,  # 0.2
        height_shift_range=0.2,  # 0.2
        horizontal_flip=True,
        vertical_flip = True)
    
    datagen.fit(x_train)
    
    flag = config3.flag()
    layers = config3.nn2()
    
    model = keras_ipi.block_builder.stack(layers)
    
    print(model.summary())

    optimizer = {'class_name': 'adam', 'config': {'lr': flag.lr, 'beta_1': flag.beta}} #otherwise  = 'adam'
    
    # loss = 'categorical_crossentropy' # TODO find out if I can change this: loss = {'class_name': 'categorical_crossentropy', 'config' : {}}
    # keras.losses.categorical_crossentropy
    
    loss = keras_ipi.losses.weigthed_crossentropy(k = layers['k'], r = layers['r'], normalize = False)
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
    dice = keras_ipi.metrics.dice

    # keras.metrics.categorical_crossentropy

    metrics = ['accuracy', sens, prec, loss, dice]

    # todo
    model.compile(loss = loss,
                  optimizer=optimizer,
                  metrics=metrics
                  )
    
    folder_weights = '/home/lameeus/data/ghent_altar/net_weight/main3/'
    filepath = folder_weights + 'foo_weight.h5'
    
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
    
    # #"pretraining" to have a good initial guess of the gradients
    # model.optimizer.lr.set_value = flag.lr/100
    # model.optimizer.beta = 1. - (1. - flag.beta)/100.
    # model.fit(X_train, Y_train, batch_size=flag.batch_size, epochs=1, verbose = 1)
    # model.optimizer.lr.set_value = flag.lr
    
    #TODO
    if True:
        epochs = flag.epochs
        for i in range(epochs):
            print('epoch {}/{}'.format(i, epochs))
            model.fit(x_train, y_train,
                      batch_size=flag.batch_size, epochs=1,  shuffle=True,
                      verbose=1,    # how much information to show 1 much or 0, nothing
                      # class_weight= (1.0, 10.0),
                      # validation_data=(X_test[:10000], Y_test[:10000]),
                      callbacks=callbacks_list)
            
            # # fits the model on batches with real-time data augmentation:
            # model.fit_generator(datagen.flow(x_train, y_train, shuffle=True, batch_size = 100)
            # #                     ,steps_per_epoch=flag.batch_size, epochs=10,
            #                     verbose=1,    # how much information to show 1 much or 0, nothing
            #                     # class_weight= (1.0, 10.0),
            #                     # validation_data=(X_test[:10000], Y_test[:10000]),
            #                     callbacks=callbacks_list)

            score = model.evaluate(x_train[:10000], y_train[:10000], batch_size=flag.batch_size, verbose=0)
            print('Train: {}'.format(score))
            score = model.evaluate(x_test[:10000], y_test[:10000], batch_size=flag.batch_size, verbose=0)
            print('Test: {}'.format(score))

            keras_ipi.results.roc(model, x_vali[:10000], y_vali[:10000], auc_only=True) # hand
            keras_ipi.results.roc(model, x_test[:10000], y_test[:10000], auc_only=True) # Zach

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
    
    # info = lambnet.block_info.Info(model)
    # info.output_test(8, 7, set='hand')
    # info.output_test(8, 7, set='zach')
    # info.output_vis(8, 7, set = 'hand', bool_save = False, last_layer = False)
    # info.output_vis(8, 7, set = 'zach', bool_save = False, last_layer = False)
    # #
    # # keras_ipi.
    # import numpy as np
    # print(np.shape(X_train)[0])

    # keras_ipi.results.roc(model, X_train[11000:12000], Y_train[11000:12000], auc_only=False)
    keras_ipi.results.roc(model, x_vali, y_vali, auc_only = False, set ='hand')   # hand
    keras_ipi.results.roc(model, x_test, y_test, auc_only = False, set ='zach')   # Zach


if __name__ == '__main__':
    main()
    