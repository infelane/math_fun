# Trying to build some unsupervised stuff

import keras
from keras.layers import Input, Dense
from keras.models import Model
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt
import pickle
import sys, os

# own
folder_loc = '/ipi/private/lameeus/private_Documents/python/2017_February/super_res_challenge'
cmd_subfolder = os.path.realpath(folder_loc)
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
# import config_lamb
import data_net


class Flag():
    bool_prev = True
    epochs = 0


def main():
    flag = Flag()

    folder = '/ipi/private/lameeus/private_Documents/python/2017_05/'
    batch_train = pickle.load(open(folder + "batch_train.p", "rb"))
    # batch_vali = pickle.load(open(folder + "batch_vali.p", "rb"))
    batch_test = pickle.load(open(folder + "batch_test.p", "rb"))
    
    # this is the size of our encoded representations
    input_dim = 3388
    encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
    
    def builder():
        # this is our input placeholder, first layer
        input_img = Input(shape=(input_dim,))
        
        elu = lambda x: keras.activations.elu(x, 0.01)
        activity_regularizer = keras.regularizers.l1(0.) # 1.e-4
        
        # "encoded" is the encoded representation of the input
        encoded = Dense(128, activation=elu,
                        activity_regularizer=activity_regularizer
                        )(input_img)
        encoded = Dense(64, activation=elu,
                        activity_regularizer=activity_regularizer
                        )(encoded)
        encoded = Dense(encoding_dim, activation='sigmoid',
                        activity_regularizer=activity_regularizer
                        )(encoded)

        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(64, activation=elu)(encoded)
        decoded = Dense(128, activation=elu)(decoded)
        
        activation = 'linear' # 'sigmoid')
        decoded = Dense(input_dim, activation=activation)(decoded)

        # this model maps an input to its reconstruction
        autoencoder = Model(input_img, decoded)

        # this model maps an input to its encoded representation
        encoder = Model(input_img, encoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(encoding_dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = autoencoder.layers[4](encoded_input)
        decoder_layer = autoencoder.layers[5](decoder_layer)
        decoder_layer = autoencoder.layers[6](decoder_layer)
        # create the decoder model
        decoder = Model(encoded_input, decoder_layer)
        
        return autoencoder, encoder, decoder
    
    autoencoder, encoder, decoder = builder()
    
    loss = 'mean_squared_error' # 'binary_crossentropy'
    autoencoder.compile(optimizer='adadelta', loss=loss)

    from keras.datasets import mnist
    import numpy as np
    (x_train, _), (x_test, _) = mnist.load_data()

    
    n_subset = 100000 # 10000 for small subset, 100000 for all
    x_train = batch_train.x[:n_subset]
    # Y_train = batch_train.y[:n_subset]
    # X_vali = batch_vali.x[:n_subset]
    # Y_vali = batch_vali.y[:n_subset]
    x_test = batch_test.x[:n_subset]
    # Y_test = batch_test.y[:n_subset]

    # normalization

    x_train[..., 0:3] = x_train[..., 0:3] * (28.87, 57.74, 57.74) + (50.0, 0.0, 0.0)
    x_train[..., 3:6] = x_train[..., 3:6] * (28.87, 57.74, 57.74) + (50.0, 0.0, 0.0)
    x_test[..., 0:3] = x_test[..., 0:3] * (28.87, 57.74, 57.74) + (50.0, 0.0, 0.0)
    x_test[..., 3:6] = x_test[..., 3:6] * (28.87, 57.74, 57.74) + (50.0, 0.0, 0.0)
    
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
    
    print(np.max(x_test))
    print(np.max(x_train))
    
    print(np.min(x_test))
    print(np.min(x_train))
    
    filepath = 'weights_unsuper.h5'
    if flag.bool_prev:
        autoencoder.load_weights(filepath)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, verbose=0,
                                                 save_weights_only=True, period=1)
    callbacks_list = [checkpoint]
    
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
    
    def im_from_data(data):
        w = 22
        from skimage import color
        lab = (data.reshape((w, w, 7)))[ ..., 0:3]
        # lab = lab*(100., 200., 200.) - (0., 100., 100.)
        rgb = color.lab2rgb(lab)
        return rgb

    
    n_test = 10
    code = np.random.uniform(0., 1., size=(n_test, 32))
    pred = decoder.predict(code)
    
    print(np.max(decoded_imgs))
    print(np.min(decoded_imgs))
    
    pred[pred < 0. ] = 0.
    pred[pred > 255. ] = 255.
   
    
    # 1/0
    
    for i in range(n_test):
        im = im_from_data(pred[i])

        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(im/255.)
        # plt.imshow((pred[0].reshape(22, 22, 7))[:, :, 0:3])
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))


    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        im = im_from_data(x_test[i])
        plt.imshow(im)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        # plt.imshow(decoded_imgs[i].reshape(28, 28))
        
        im = im_from_data(decoded_imgs[i])
        plt.imshow(im)
        
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    main()