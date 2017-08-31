# Trying to build some unsupervised stuff

import os
import sys

import keras
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Dense, Conv2D
from keras.models import Model

# own
folder_loc = '/ipi/private/lameeus/private_Documents/python/2017_February/super_res_challenge'
cmd_subfolder = os.path.realpath(folder_loc)
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
# import config_lamb
import keras_ipi


class Flag():
    bool_prev = True
    epochs = 10
    lr = 1e-4


def main():
    flag = Flag()

    folder = '/ipi/private/lameeus/private_Documents/python/2017_05/'
    folder = '/home/lameeus/data/ghent_altar/input_arrays/'
    # batch_train = pickle.load(open(folder + "batch_train.p", "rb"))
    # batch_vali = pickle.load(open(folder + "batch_vali.p", "rb"))
    # batch_test = pickle.load(open(folder + "batch_test.p", "rb"))

    xy = np.load(folder + 'xy_hand_ext7.npz')
    xx = xy['x']
    yy = xy['y']
    batch_train = [xx, yy]
    
    # this is the size of our encoded representations
    w = 21  # 21
    input_dim = w*w*7     # 22*22*7
    encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
    
    def builder():
        elu = lambda x: keras.activations.elu(x, 0.01)
        activity_regularizer = keras.regularizers.l1(0.) # 1.e-4
        
        # (possible) inputs
        # this is our input placeholder, first layer
        input_img = Input(shape=(input_dim,))
        # create a placeholder for an encoded (32-dimensional) input
        if 0:
            middle = Input(shape=(encoding_dim,))
        elif 0:
            middle = Input(shape=(21, 21, encoding_dim))
        else:
            middle = Input(shape=(19, 19, encoding_dim))
        
        if 0:
            # "encoded" is the encoded representation of the input
            trainable = False
            encoded = Dense(128, activation=elu,
                            activity_regularizer=activity_regularizer,
                            trainable= trainable
                            )(input_img)
            encoded = Dense(64, activation=elu,
                            activity_regularizer=activity_regularizer,
                            trainable= trainable
                            )(encoded)
            encoded = Dense(encoding_dim, activation='sigmoid',
                            # activity_regularizer=activity_regularizer,
                            trainable= trainable
                            )(encoded)
    
            # "decoded" is the lossy reconstruction of the input
            decoded = Dense(64, activation=elu,trainable= trainable
                            )(middle)
            # decoded = Dense(64, activation=elu)(encoded)
            decoded = Dense(128, activation=elu, trainable= trainable
                            )(decoded)
            
            activation = 'linear' # 'sigmoid')
            decoded = Dense(input_dim, activation=activation, trainable= trainable
                            )(decoded)
            
        if 0:
            # "encoded" is the encoded representation of the input
            encoded = Dense(encoding_dim, activation=elu,
                            activity_regularizer=activity_regularizer
                            )(input_img)

            decoded = Dense(input_dim, activation='sigmoid')(middle)
            
        if 1:
            # "encoded" is the encoded representation of the input
            encoded1 = keras.layers.Reshape((21, 21, 7))(input_img)
            encoded2 = Conv2D(encoding_dim, (3,3), activation='sigmoid',
                            activity_regularizer=activity_regularizer,
                            padding='valid'
                            )(encoded1)
            encoded = keras.layers.GaussianNoise(stddev=1.)(encoded2)
            
            decoded = keras.layers.Conv2DTranspose(7, (3,3), activation='sigmoid', padding = 'valid')(middle)
            decoded = keras.layers.Flatten()(decoded)

        # this model maps an input to its encoded representation
        encoder = Model(input_img, encoded)

        # create the decoder model
        decoder = Model(middle, decoded)

        # this model maps an input to its reconstruction
        autoencoder = Model(input_img, decoder(encoder.output))



        # # create a placeholder for an encoded (32-dimensional) input
        # encoded_input = Input(shape=(encoding_dim,))
        # retrieve the last layer of the autoencoder model
        # decoder_layer = autoencoder.layers[4](encoded_input)
        # decoder_layer = autoencoder.layers[5](decoder_layer)
        # decoder_layer = autoencoder.layers[6](decoder_layer)
        # # create the decoder model
        # decoder = Model(encoded_input, decoder_layer)
        
        return autoencoder, encoder, decoder
    
    autoencoder, encoder, decoder = builder()
    
    loss = 'mean_squared_error' # 'binary_crossentropy'
    adam = keras.optimizers.adam(lr = flag.lr)
    
    psnr = keras_ipi.metrics.psnr
    
    autoencoder.compile(optimizer=adam, loss=loss, metrics=[psnr])

    from keras.datasets import mnist
    (x_train, _), (x_test, _) = mnist.load_data()

    
    # n_subset = 100000 # 10000 for small subset, 100000 for all
    
    n_all = len(batch_train[0])
    n_subset = int(0.8*n_all)
    
    x_train = batch_train[0][:n_subset]
    # Y_train = batch_train.y[:n_subset]
    # X_vali = batch_vali.x[:n_subset]
    # Y_vali = batch_vali.y[:n_subset]
    x_test = batch_train[0][n_subset:]
    # Y_test = batch_test.y[:n_subset]
    
    # x_train[..., 0:3] = (x_train[..., 0:3]/(2*np.sqrt(3.))) + (0.5, 0., 0.)
    # x_train[..., 3:6] = (x_train[..., 3:6]/(2*np.sqrt(3.))) + (0.5, 0., 0.)
    # x_test[..., 0:3] = (x_test[..., 0:3]/(2*np.sqrt(3.))) +  (0.5, 0., 0.)
    # x_test[..., 3:6] = (x_test[..., 3:6]/(2*np.sqrt(3.))) +  (0.5, 0., 0.)

    # axis = [0, 1, 2]
    # axis = 0
    # print(np.max(x_train.reshape((np.prod(x_train.shape[0:3]), 7)), axis = 1))
    # print(np.mean(x_train.reshape((np.prod(x_train.shape[0:3]), 7)), axis = 1))
    # print(np.min(x_train.reshape((np.prod(x_train.shape[0:3]), 7)), axis = 1))
    # print(np.mean(x_train, axis=axis))
    # print(np.min(x_train, axis=axis))

    # x_train = x_train.astype('float32') / 255.
    # x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    # print(x_train.shape)
    # print(x_test.shape)
    
    folder = '/home/lameeus/data/NO_PLACE/'
    filepath = 'weights_unsuper.h5'
    if flag.bool_prev:
        autoencoder.load_weights(folder + filepath)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, verbose=0,
                                                 save_weights_only=True, period=1)
    callbacks_list = [checkpoint]
    
    if 0:
        autoencoder.fit(x_train, x_train,
                        epochs=flag.epochs,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(x_test, x_test),
                        callbacks = callbacks_list)

    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_imgs = encoder.predict(x_test[0:1000])
    
    print(encoded_imgs[0, :])
    
    decoded_imgs = decoder.predict(encoded_imgs)
    
    # def im_from_data(data):
        # from skimage import color
        # lab = (data.reshape((w, w, 7)))[ ..., 0:3]
        # lab = lab*(100., 200., 200.) - (0., 100., 100.)
        
        # rgb = color.lab2rgb(lab)
        # return rgb
    
    def im_from_data2(data):
        return np.reshape(data, (w,w,7))[..., 0:3]
    
    n_test = 10
    code = np.random.uniform(0., 1., size=(n_test, 19, 19, 32))
    pred = decoder.predict(code)
    
    pred[pred < 0. ] = 0.
    pred[pred > 1. ] = 1.

    # 1/0
    if 0:
        for i in range(n_test):
            im = im_from_data2(pred[i])
            
            ax = plt.subplot(2, 10, i + 1)
            plt.imshow(im)
            # plt.imshow((pred[0].reshape(22, 22, 7))[:, :, 0:3])
            # plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))

    idx = np.arange(999)
    np.random.seed(3)
    np.random.shuffle(idx)

    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        im = im_from_data2(x_test[idx[i]])
        plt.imshow(im)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n)
        im = (encoded_imgs[idx[i]])[..., 0:3]
        plt.imshow(im)
        
        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + 2*n)
        # plt.imshow(decoded_imgs[i].reshape(28, 28))
        
        im = im_from_data2(decoded_imgs[idx[i]])
        plt.imshow(im)
        
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    main()
