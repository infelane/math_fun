import keras
from keras.layers import Conv2D, Conv2DTranspose, Input
from keras.models import Model
from keras.losses import mean_squared_error
import keras_ipi
from maus.paint_tools import image_tools
import matplotlib.pyplot as plt
import numpy as np


def gen_encoder(layer_in):
    x = Conv2D(5, (1,1), activation='sigmoid', padding='valid')(layer_in)
    return Model(layer_in, x)
    
def gen_decoder(layer_code):
    x = Conv2DTranspose(3, (1, 1), activation = 'sigmoid', padding = 'valid')(layer_code)
    return Model(layer_code ,x)

def show_images(im1, im2):
    plt.subplot(2, 1, 1)
    plt.imshow(im1)
    plt.subplot(2, 1, 2)
    plt.imshow(im2)
    plt.show()

def main():
    path = '/home/lameeus/data/ghent_altar/input/19_clean.tif'
    im1 = image_tools.path2im(path)
    
    path = '/home/lameeus/data/ghent_altar/input/13_new_clean_reg1.tif'
    im2 = image_tools.path2im(path)
    
    if 0:
        show_images(im1, im2)

    width = 10
    shape_input = (width, width, 3)
    shape_code = (width, width, 5)
    
    layer_in1 = Input(shape_input, name='x_in')
    layer_in2 = Input(shape_input, name='x_in2')
    layer_code1 = Input(shape_code, name='x_code')
    layer_code2 = Input(shape_code, name='x_code2')
    
    layer_in1_all = Input(np.shape(im1), name = 'x_im1')
    layer_in2_all = Input(np.shape(im2), name = 'x_im2')
        
    encoder1 = gen_encoder(layer_in1)
    encoder2 = gen_encoder(layer_in2)
    decoder1 = gen_decoder(layer_code1)
    decoder2 = gen_decoder(layer_code2)

    star_1 = encoder1(layer_in1)
    star_2 = encoder2(layer_in2)
    x_11 = decoder1(star_1)
    x_22 = decoder2(encoder2(layer_in2))
    x_121 = decoder1(encoder2(decoder2(star_1)))
    x_212 = decoder2(encoder1(decoder1(star_2)))
    
    x_11_all = decoder1(encoder1(layer_in1_all))
    x_12_all = decoder2(encoder1(layer_in1_all))
    x_121_all = decoder1(encoder2(x_12_all))
    x_22_all = decoder2(encoder2(layer_in2_all))
    x_21_all = decoder1(encoder2(layer_in2_all))
    x_212_all = decoder2(encoder1(x_21_all))

    inputs = [layer_in1, layer_in2]
    outputs = [x_11, x_22, x_121, x_212]
    
    adam = keras.optimizers.adam(lr = 1e-3)
    
    model = Model(inputs = inputs , outputs = outputs)
    model.compile(optimizer=adam,
                  loss=mean_squared_error
                  )
    
    model1 = Model(inputs = layer_in1_all, outputs = [x_11_all, x_12_all, x_121_all])
    model2 = Model(inputs=layer_in2_all, outputs=[x_22_all, x_21_all, x_212_all])
        
    shape = np.shape(im1)
    shape2 = np.shape(im2)
    n_train = 2000
    h_random = np.random.randint(0, shape[0] - width, (n_train, ))
    w_random = np.random.randint(0, shape[1] - width, (n_train, ))
    h_random2 = np.random.randint(0, shape2[0] - width, (n_train, ))
    w_random2 = np.random.randint(0, shape2[1] - width, (n_train, ))

    x_train1 = []
    x_train2 = []
    for i in range(n_train):
        h_0 = h_random[i]
        h_1 = h_0 + width
        w_0 = w_random[i]
        w_1 = w_0 + width

        h_2 = h_random2[i]
        w_2 = w_random2[i]
        
        print(h_0)
        
        x_train1.append(im1[h_0:h_1, w_0:w_1, ...])
        x_train2.append(im2[h_2:h_2 + width, w_2:w_2+width, ...])

    x_train1 = np.array(x_train1)
    x_train2 = np.array(x_train2)
    
    def save_submodels():
        folder_model = '/home/lameeus/data/ghent_altar/net_weight/gan_lamb/'
        name = 'encoder1.h5'
        encoder1.save_weights(folder_model + name)
        name = 'encoder2.h5'
        encoder2.save_weights(folder_model + name)
        name = 'decoder1.h5'
        decoder1.save_weights(folder_model + name)
        name = 'decoder2.h5'
        decoder2.save_weights(folder_model + name)

    def load_submodels():
        folder_model = '/home/lameeus/data/ghent_altar/net_weight/gan_lamb/'
        name = 'encoder1.h5'
        encoder1.load_weights(folder_model + name)
        name = 'encoder2.h5'
        encoder2.load_weights(folder_model + name)
        name = 'decoder1.h5'
        decoder1.load_weights(folder_model + name)
        name = 'decoder2.h5'
        decoder2.load_weights(folder_model + name)
        
    if 1:
        load_submodels()

    n_epochs = 100
    
    if n_epochs:
        model.fit([x_train1, x_train2], [x_train1, x_train2, x_train1, x_train2], epochs = n_epochs)
        save_submodels()
    
    x_im1 = np.stack([im1], axis = 0)
    x_im2 = np.stack([im2], axis=0)
    
    y_im1 = model1.predict(x_im1)
    y_im2 = model2.predict(x_im2)
    
    plt.subplot(2, 3, 1)
    plt.imshow(y_im1[0][0, ...])
    plt.subplot(2, 3, 2)
    plt.imshow(y_im1[1][0, ...])
    plt.subplot(2, 3, 3)
    plt.imshow(y_im1[2][0, ...])
    plt.subplot(2, 3, 4)
    plt.imshow(y_im2[0][0, ...])
    plt.subplot(2, 3, 5)
    plt.imshow(y_im2[1][0, ...])
    plt.subplot(2, 3, 6)
    plt.imshow(y_im2[2][0, ...])
    plt.show()

    
if __name__ == '__main__':
    main()