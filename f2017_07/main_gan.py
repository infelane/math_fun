""" Try to run some GAN stuff
idea: Convert 0's to 5's e.g."""

import numpy as np
import keras
from keras.layers import Conv2D, Input, Conv2DTranspose, MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.models import Model
from keras.metrics import binary_accuracy
import matplotlib.pyplot as plt
from keras.optimizers import adam
import time

from keras.callbacks import TensorBoard

# My packages
from keras_ipi.metrics import psnr, norm_cost
from keras_ipi.layers import CroppDepth
import keras_ipi


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
        
        
def plotter(data):
    plt.imshow(data, interpolation='nearest', cmap='Greys', vmin = 0, vmax = 1)


def time_func(func, n = 1):
    start = time.time()
    for i in range(n):
        a = func()
    end = time.time()
    print('elapsed time: {} s'.format(end - start))
    return a


def main():
    """ settings """
    n_digits = 10
    
    """ compiling """
    loss = keras.losses.mean_squared_error
    lr_auto = 1e-4 # 1.e-4
    lr_discr = 1.e-4
    lr_super = 1.e-5 # 1.e-4
    
    """ start """
    
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    def add_axis(data):
        return np.stack([data], axis = 3)

    n_train = 6000  # todo 6000
    x_train = x_train[:n_train, ...]
    y_train = y_train[:n_train, ...]
    
    x_train = add_axis(x_train)
    x_test = add_axis(x_test)
    
    def norm(data):
        max_data = np.max(data)
        if max_data <= 1.:
            return data
        elif max_data <= 255:
            return data/255
        elif max_data <= 65535:
            return data/65535
    
    x_train = norm(x_train)
    x_test = norm(x_test)
    y_train_cat = np.reshape(keras.utils.to_categorical(y_train, n_digits + 1), newshape=(-1, 1, 1, n_digits + 1))
    y_test_cat = np.reshape(keras.utils.to_categorical(y_test, n_digits + 1), newshape=(-1, 1, 1, n_digits + 1))
    
    ind_0 = np.equal(y_train, 0)
    ind_1 = np.equal(y_train, 1)
    ind_0_test = np.equal(y_test, 0)
    ind_1_test = np.equal(y_test, 1)
    
    x_0 = x_train[ind_0, ...]
    x_1 = x_train[ind_1, ...]

    x_i = {}
    x_i_test = {}
    for i in range(n_digits):
        ind_i = np.equal(y_train, i)
        ind_i_test = np.equal(y_test, i)
        x_i.update({i: x_train[ind_i, ...]})
        x_i_test.update({i: x_test[ind_i_test, ...]})
    
    x_0_test = x_test[ind_0_test, ...]
    x_1_test = x_test[ind_1_test, ...]
    
    print(np.shape(x_0))
    
    n_code = 50
    
    """ the model(s) """
    auto_1 = 0
    auto_2 = 1
    
    shape_input = (28, 28, 1)
    if auto_1:
        shape_code = (4, 4, 8)
    elif auto_2:
        shape_code = (1, 1, n_code)
    
    layer_in = Input(shape_input, name='x_in')
    # input_1 = Input(shape_input, name='x_1')
    
    layer_code = Input(shape_code, name='x_code')
    output_p = Input((1, 1, 10), name = 'd_code')
    
    def model_g_in_star(i):
        def layers_g_in_star():
            activation = 'elu'
            if auto_1:
                x = Conv2D(16, (3, 3), activation=activation, padding='same')(layer_in)
                x = MaxPooling2D((2, 2), padding='same')(x)
                x = Conv2D(8, (3, 3), activation=activation, padding='same')(x)
                x = MaxPooling2D((2, 2), padding='same')(x)
                x = Conv2D(8, (3, 3), activation='sigmoid', padding='same')(x)
                x = MaxPooling2D((2, 2), padding='same')(x)
                
            elif auto_2:
                x = Conv2D(16, (3, 3), activation=activation, padding='valid')(layer_in)
                x = AveragePooling2D((2, 2), padding='same')(x)
                x = Conv2D(8, (4, 4), activation=activation, padding='valid')(x)
                x = AveragePooling2D((2, 2), padding='same')(x)
                x = Conv2D(8, (5, 5), activation=activation, padding='valid')(x)
                x = Conv2D(n_code, (1, 1), activation='sigmoid', padding='valid')(x)
                
                

                # x = Conv2D(n_code, (28, 28), name='l_in_star_0', activation='sigmoid')(layer_in)
            return x
            
        
        layer_g_in_star_i = layers_g_in_star()
        return Model(layer_in, layer_g_in_star_i, name = 'mod_in_star{}'.format(i))
    
    def model_g_star_out(i):
        def layers_g_star_out():
            activation = 'elu'
            if auto_1:
                x = Conv2D(8, (3, 3), activation=activation, padding='same')(layer_code)
                x = UpSampling2D((2, 2))(x)
                x = Conv2D(8, (3, 3), activation=activation, padding='same')(x)
                x = UpSampling2D((2, 2))(x)
                x = Conv2D(16, (3, 3), activation=activation)(x)
                x = UpSampling2D((2, 2))(x)
                x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
            elif auto_2:
                x = Conv2DTranspose(8, (5, 5), activation=activation, padding='valid')(layer_code)
                x = UpSampling2D((2, 2))(x)
                x = Conv2DTranspose(8, (4, 4), activation=activation, padding='valid')(x)
                x = UpSampling2D((2, 2))(x)
                x = Conv2DTranspose(16, (3, 3), activation=activation, padding='valid')(x)
                x = Conv2D(1, (1, 1), activation='sigmoid', padding='valid')(x)
                
                # x = Conv2DTranspose(1, (28, 28), name='l_star_out_0', activation='sigmoid')(layer_code)
            return x
            
        layer_g_star_out_i = layers_g_star_out()
        return Model(layer_code, layer_g_star_out_i, name= 'mod_star_out{}'.format(i))
    
    def model_d_im_p():
        activation = 'elu'
        layer_d_im_code = Conv2D(25, (5, 5), name='d_im_code0', activation=activation)(layer_in)
        layer_d_im_code = MaxPooling2D((2, 2))(layer_d_im_code)
        layer_d_im_code = Conv2D(25, (5, 5), name = 'd_im_code1', activation=activation)(layer_d_im_code)
        layer_d_im_code = MaxPooling2D((2, 2))(layer_d_im_code)
        layer_d_im_code = Conv2D(50, (4, 4), name = 'd_im_code2', activation=activation)(layer_d_im_code)
        # the 11th will be for "everything else"
        layer_d_code_p = Conv2D(n_digits + 1, (1, 1), name='d_code_p', activation='softmax')(layer_d_im_code)
        
        return Model(layer_in, layer_d_code_p, name = 'mod_g_in_p')
    
    def model_d_pi(i):
        layer_d_pi = CroppDepth((i ,i+1), name = 'd_pi')(output_p)
        return Model(output_p, layer_d_pi)
        
    def stack_models(model1, model2):
        # by reusing, you set the inputs from model2 again => might add
        # this can give problems:   return Model(model1.input, model2(model1.output))
        return Model(model1.get_input_at(0), model2(model1.get_output_at(0)))
    
    model_d_im_code = model_d_im_p()
    model_d_im_code.compile(optimizer=keras.optimizers.adam(lr=lr_discr), loss = loss, metrics=[binary_accuracy])
    
    dict_model_g_in_star = {}
    dict_model_g_star_out = {}
    dict_model_g = {}
    dict_model_g_trans = {}
    dict_model_d_i = {}
    dict_model_super = {}
    dict_model_auto = {} # only the auto-encoders with randomness
    
    for i in range(n_digits):
        dict_model_g_in_star.update({i: model_g_in_star(i)})
        dict_model_g_star_out.update({i: model_g_star_out(i)})
    
    for dict_i in dict_model_g_in_star:
        for dict_j in dict_model_g_star_out:
            # Combine all the coders/encoders
            
            model =  stack_models(dict_model_g_in_star[dict_i], dict_model_g_star_out[dict_j])
            # noise_star = keras.layers.noise.GaussianNoise(0.1, trainable=False)  # to maximize entropy in encoding
            # model =  dict_model_g_star_out[dict_j](dict_model_g_in_star[dict_i])
            dict_model_g.update({(dict_i, dict_j):model})

    # TODO get to work, might have to compile layers?
    for dict_i in dict_model_g_in_star:
        noise_star = keras.layers.noise.GaussianNoise(0.05, trainable=False)  # to maximize entropy in encoding
        model = stack_models(dict_model_g_in_star[dict_i], noise_star)
        model = stack_models(model, dict_model_g_star_out[dict_i])
        # model = dict_model_g_star_out[dict_i](noise_star(dict_model_g_in_star[dict_i]))

        make_trainable(model, True)
        model.compile(optimizer=keras.optimizers.adam(lr=lr_auto), loss=norm_cost)  # the norm cost is l2, but normalized

        dict_model_auto.update({dict_i : model})

    for dict_i in dict_model_g_in_star:
        for dict_j in dict_model_g_star_out:
            # translation through other number
            model1 = dict_model_g[(dict_i, dict_j)]
            model2 = dict_model_g[(dict_j, dict_i)]

            dict_model_g_trans.update({(dict_i, dict_j): stack_models(model1, model2)})
            
    for i in range(n_digits):
        modeli = model_d_pi(i)
        # modeli = modeli(model_d_im_code)
        modeli = stack_models(model_d_im_code, modeli)
        modeli.trainable = True
        modeli.compile(optimizer=keras.optimizers.adam(lr=lr_discr), loss=loss, metrics=[binary_accuracy])
        dict_model_d_i.update({i: modeli})

    # model_d_im_p_0 = stack_models(model_d_im_code, model_d_p_0)
    # model_d_im_p_1 = stack_models(model_d_im_code, model_d_p_1)
    
    folder_model = '/home/lameeus/data/ghent_altar/net_weight/gan_simple/'

    def layer_subtract(a, b, name=None):
        b_ = keras.layers.Lambda(lambda x: -x, trainable= False, name = 'neg_' + name)(b)
        # b_ = lambda(lambda b : -b)(b)
        return keras.layers.Add(name=name, trainable= False)([a, b_])
    
    def layer_1_min(a, name = None):
        return keras.layers.Lambda(lambda x: 1 - x, name = name, trainable=False)(a)
    
    def layer_name(a, name = None):
        return keras.layers.Lambda(lambda x: x, name = name, trainable=False)(a)
    
    # def super_model(i = -1):
    #     # input_i = layer_in
    #     # layer_star = dict_model_g_in_star[i]
    #     # g_ii = dict_model_g[(i, i)](layer_in)
    #
    #     g_i_star = dict_model_g_in_star[(i)](layer_in)
    #
    #     g_ii = dict_model_g_star_out[(i)](g_i_star)
    #
    #     out_name_self = 'out_self{}'.format(i)
    #     res_ii = layer_subtract(g_ii, layer_in, out_name_self)
    #     name_self_d = 'd_self{}'.format(i)
    #     # d_ii = layer_name(dict_model_d_i[(i)](g_ii), name_self_d)
    #
    #     d_i = dict_model_d_i[(i)]
    #     make_trainable(d_i, False)
    #     d_ii = layer_1_min(d_i(g_ii), name_self_d)
    #
    #     outputs = [res_ii, d_ii]
    #     dict_loss = {out_name_self : loss, name_self_d : loss}
    #     dict_weights = {out_name_self : 1., name_self_d : 0.00001}
    #
    #     for j in range(n_digits):
    #         if j != i:
    #             g_ij = dict_model_g_star_out[j](g_i_star)
    #
    #             g_ijstar = dict_model_g_in_star[j](g_ij)
    #
    #             g_iji = dict_model_g_star_out[(i)](g_ijstar)
    #             g_ijj = dict_model_g_star_out[(j)](g_ijstar)
    #
    #             name_auto_iji = 'res_{}-{}-{}'.format(i, j, i)
    #             name_auto_ijj = 'res_{}-{}-{}'.format(i, j, j)
    #             name_d_j = 'out_d{}'.format(j)
    #
    #             d_j = dict_model_d_i[j]
    #             make_trainable(d_j, False)
    #
    #             res_iji = layer_subtract(g_iji, layer_in, name_auto_iji)
    #             res_ijj = layer_subtract(g_ijj, g_ij, name_auto_ijj)
    #             # gd_ij = layer_name(d_j(g_ij), name=name_d_j)
    #             gd_ij = layer_1_min(d_j(g_ij), name = name_d_j)
    #
    #             outputs.extend([res_iji, res_ijj, gd_ij])
    #             dict_loss.update({name_auto_iji: loss, name_auto_ijj: loss, name_d_j: loss})
    #             dict_weights.update({name_auto_iji: 0.000001, name_auto_ijj: 0.000001, name_d_j: 0.00001})
    #
    #     superr = Model(inputs =layer_in, outputs = outputs)
    #     superr.compile(optimizer=adam(lr_super),
    #                    loss = dict_loss,
    #                    loss_weights=dict_weights,
    #                    )
    #
    #     return superr
    
    def super_model2(i = -1):
        name_auto = 'auto_{}_{}'.format(i, i)
        name_self_d = 'd_{}_{}'.format(i, i)
        name_stars = 'auto_stars{}'.format(i)
        
        g_i_star = dict_model_g_in_star[(i)]
        g_star_i = dict_model_g_star_out[(i)]
        
        make_trainable(g_i_star, True)
        make_trainable(g_star_i, True)
        
        # the "model" of the first *
        noise_star = keras.layers.noise.GaussianNoise(0.5, trainable= False) # to maximize entropy in encoding
        star0 = g_i_star(layer_in)
        star0_noise = noise_star(star0)
        x_ii = layer_name(g_star_i(star0_noise), name_auto)
        
        d_i = dict_model_d_i[(i)]
        make_trainable(d_i, False)
        
        # res_ii = layer_subtract(x_ii, layer_in, out_name_self)
        d_ii = layer_1_min(d_i(x_ii), name_self_d)
        
        star1 = g_i_star(x_ii)
        res_stars = layer_subtract(star0, star1, name_stars)
        
        # outputs = [res_ii, d_ii, res_stars]
        outputs = [x_ii, d_ii, res_stars]
        dict_loss = {name_auto : norm_cost, name_self_d : loss, name_stars : loss}
        # dict_weights = {out_name_self : 1., name_self_d : 0.001}
        dict_weights = {name_auto : 1., name_self_d : 0.1, name_stars : 0.1}
        
        for j in range(n_digits):
            if j != i:
                name_d_j = 'out_d{}'.format(j)
                name_auto_iji = 'auto_{}-{}-{}'.format(i, j, i)
                name_auto_ijj = 'auto_{}-{}-{}'.format(i, j, j)
                
                g_starj = dict_model_g_star_out[j]
                g_jstar = dict_model_g_in_star[j]
                
                make_trainable(g_starj, False)
                make_trainable(g_jstar, False)

                x_ij = g_starj(star0)
                
                # noise_star = keras.layers.noise.GaussianNoise(0.1, trainable=False)  # to maximize entropy in encoding
                star2 = g_jstar(x_ij)
                # star2_noise = noise_star(star2)
                
                d_j = dict_model_d_i[j]
                make_trainable(d_j, False)

                gd_ij = layer_1_min(d_j(x_ij), name=name_d_j)

                # outputs.extend([gd_ij])
                # dict_loss.update({name_d_j: loss})
                # dict_weights.update({name_d_j: 0.0001})

                x_iji = layer_name(g_star_i(star2), name_auto_iji)
                # x_ijj = g_starj(star2)
                
                # g_ijj = dict_model_g_star_out[(j)](g_ijstar)
                # res_iji = layer_subtract(x_iji, layer_in, name_auto_iji)
                # res_ijj = layer_subtract(x_ijj, x_ij, name_auto_ijj)
                # outputs.extend([res_iji, res_ijj, gd_ij])
                # outputs.extend([x_iji, res_ijj, gd_ij])
                outputs.extend([x_iji, gd_ij])
                # dict_loss.update({name_auto_iji: norm_cost, name_auto_ijj: loss, name_d_j: loss})
                dict_loss.update({name_auto_iji: norm_cost, name_d_j: loss})
                # dict_weights.update({name_auto_iji: 0.00001, name_auto_ijj: 0.00001, name_d_j: 0.0001})
                # dict_weights.update({name_auto_iji: 0.01, name_auto_ijj: 0, name_d_j: 0.001})
                # dict_weights.update({name_auto_iji: 0.1, name_auto_ijj: 0, name_d_j: 0.01})
                dict_weights.update({name_auto_iji: 0.1, name_d_j: 0.01})
                

            

        superr = Model(inputs=layer_in, outputs=outputs)

        print("--------\n")
        for l in superr.layers:
            print("{} {}".format(l.name, l.trainable))
        #     l.trainable = val
        
        for l in superr.layers:
            if l.name == 'model_208':
                for ll in l.layers:
                    print("{} {}".format(ll.name, ll.trainable))
        
        superr.compile(optimizer=adam(lr_super),
                       loss=dict_loss,
                       loss_weights=dict_weights,
                       )
        
        return superr

    def super_deluxe_model():
        # TODO train everything at the same time
    
        layer_in = []
        outputs = []
        dict_loss = {}
        dict_weights = {}
        for i in range(n_digits):
            name_input = 'x_in_{}'.format(i)
            name_auto_ii = 'x_{}{}'.format(i, i)
            name_res_ii = 'res_{}{}'.format(i, i)
            
            layer_i = Input(shape_input, name=name_input)
            layer_in.append(layer_i)

            g_i_star = dict_model_g_in_star[(i)]
            g_star_i = dict_model_g_star_out[(i)]
            make_trainable(g_i_star, True)
            make_trainable(g_star_i, True)

            star0_i = g_i_star(layer_i)
            x_ii = layer_name(g_star_i(star0_i), name_auto_ii)
            res_ii = layer_subtract(layer_i, x_ii, name=name_res_ii)

            # outputs.append(res_ii)
            # dict_loss.update({name_res_ii: loss})
            # dict_weights.update({name_res_ii: 1.})
            outputs.append(x_ii)
            dict_loss.update({name_auto_ii: norm_cost})
            dict_weights.update({name_auto_ii: 1.})
            
            for j in range(n_digits):
                if j != i:
                    name_x_iji = 'x_{}{}{}'.format(i, j, i)
                    name_res_iji = 'res_{}{}{}'.format(i, j, i)
                    name_res_star = 'res_star_{}{}'.format(i, j)
                    name_d_j = 'discr_{}{}'.format(i, j)
                    
                    g_j_star = dict_model_g_in_star[(j)]
                    g_star_j = dict_model_g_star_out[(j)]
                    make_trainable(g_j_star, True)
                    make_trainable(g_star_j, True)
                    
                    x_ij = g_star_j(star0_i)
                    star1_ij = g_j_star(x_ij)
                    x_iji = layer_name(g_star_i(star1_ij), name_x_iji)

                    res_iji = layer_subtract(layer_i, x_iji, name=name_res_iji)
                    res_star = layer_subtract(star0_i, star1_ij, name = name_res_star)
                    
                    d_j = dict_model_d_i[j]
                    make_trainable(d_j, False)

                    gd_ij = layer_1_min(d_j(x_ij), name=name_d_j)

                    # outputs.extend([res_iji, res_star, gd_ij])
                    # dict_loss.update({name_res_iji: loss, name_res_star : loss, name_d_j: loss})
                    # dict_weights.update({name_res_iji: 1., name_res_star : 1., name_d_j : 1.})
                    outputs.extend([x_iji, res_star, gd_ij])
                    dict_loss.update({name_x_iji: norm_cost, name_res_star: loss, name_d_j: loss})
                    dict_weights.update({name_x_iji: 1., name_res_star: 1., name_d_j: 1.})

        superr = Model(inputs=layer_in, outputs=outputs)
        
        superr.compile(optimizer=adam(lr_super),
                       loss=dict_loss,
                       loss_weights=dict_weights,
                       )
            
        return superr

    for dict_i in dict_model_g:
        if dict_i[0] == dict_i[1]:
            foo = dict_model_g[dict_i]
            make_trainable(foo, True)
            foo.compile(optimizer=keras.optimizers.adam(lr=lr_auto), loss=loss, metrics=[norm_cost])

    if 0:
        for i in range(n_digits):
            # dict_model_super.update({i : super_model(i)})
            # TODO switched
            # TODO
            # TODO
            dict_model_super.update({i: super_model2(i)})

    model_super_deluxe = super_deluxe_model()

    # for dict_i in dict_model_g_trans:
    #     dict_model_g_trans[dict_i].compile(optimizer=keras.optimizers.adam(lr = lr_auto), loss=loss, metrics=[psnr])
    #

    def load_submodels():
        for i in range(n_digits):        # if i != 1 and i != 2:
            name = 'w_{}_star.h5'.format(i)
            dict_model_g_in_star[i].load_weights(folder_model + name)
            name = 'w_star_{}.h5'.format(i)
            dict_model_g_star_out[i].load_weights(folder_model + name)

        model_d_im_code.load_weights(folder_model + 'w_discr.h5')

    def save_submodels():
        for i in range(n_digits):
            name =  'w_{}_star.h5'.format(i)
            dict_model_g_in_star[i].save_weights(folder_model + name)
            name =  'w_star_{}.h5'.format(i)
            dict_model_g_star_out[i].save_weights(folder_model + name)
            
        model_d_im_code.save_weights(folder_model + 'w_discr.h5')
        
    if 1:
        load_submodels()
        
    # def testing():
    #     n_testing = 10
    #
    #     x_orig_0 = x_0_test[:n_testing, ...]
    #     x_orig_1 = x_1_test[:n_testing, ...]
    #
    #     x_pred_0_1 = dict_model_g[(0, 1)].predict(x_orig_0)
    #     x_pred_0_0 = dict_model_g[(0, 0)].predict(x_orig_0)
    #     x_pred_0_1_0 = dict_model_g_trans[(0, 1)].predict(x_orig_0)
    #
    #     x_pred_1_0 = dict_model_g[(1, 0)].predict(x_orig_1)
    #     x_pred_1_1 = dict_model_g[(1, 1)].predict(x_orig_1)
    #     x_pred_1_0_1 = dict_model_g_trans[(1, 0)].predict(x_orig_1)
    #
    #     def row_start(row):
    #         return n_testing*row + 1
    #
    #
    #     n_row = 8
    #     for i in range(n_testing):
    #         row = 0
    #         plt.subplot(n_row,n_testing,row_start(row)+ i)
    #         plotter(x_orig_0[i, ..., 0])
    #         plt.title('0 orig')
    #
    #         row = 1
    #         plt.subplot(n_row,n_testing,row_start(row) + i)
    #         plotter(x_pred_0_0[i, ..., 0])
    #         plt.title('0 to 0')
    #
    #         row = 2
    #         plt.subplot(n_row,n_testing,row_start(row) + i)
    #         plotter(x_pred_0_1_0[i, ..., 0])
    #         plt.title('0 to 1 to 0')
    #
    #         row = 3
    #         plt.subplot(n_row,n_testing,row_start(row) + i)
    #         plotter(x_pred_0_1[i, ..., 0])
    #         plt.title('0 to 1')
    #
    #         row = 4
    #         plt.subplot(n_row, n_testing, row_start(row) + i)
    #         plotter(x_orig_1[i, ..., 0])
    #         plt.title('1 orig')
    #
    #         row = 5
    #         plt.subplot(n_row, n_testing, row_start(row) + i)
    #         plotter(x_pred_1_1[i, ..., 0])
    #         plt.title('1 to 1')
    #
    #         row = 6
    #         plt.subplot(n_row, n_testing, row_start(row) + i)
    #         plotter(x_pred_1_0_1[i, ..., 0])
    #         plt.title('1 to 0 to 1')
    #
    #         row = 7
    #         plt.subplot(n_row, n_testing, row_start(row) + i)
    #         plotter(x_pred_1_0[i, ..., 0])
    #         plt.title('1 to 0')
    #
    #     plt.show()
        
    def testing(digit_in = 0, digit_out = 1):
        n_testing = 10
        x_orig_i = x_i_test[digit_in][:n_testing, ...]
        x_pred_i_j = dict_model_g[(digit_in, digit_out)].predict(x_orig_i)
        x_pred_i_i = dict_model_g[(digit_in, digit_in)].predict(x_orig_i)
        x_pred_i_j_i = dict_model_g_trans[(digit_in, digit_out)].predict(x_orig_i)

        def row_start(row):
            return n_testing * row + 1

        n_row = 4
        for i in range(n_testing):
            row = 0
            plt.subplot(n_row, n_testing, row_start(row) + i)
            plotter(x_orig_i[i, ..., 0])
            plt.title('{} orig'.format(digit_in))
    
            row = 1
            plt.subplot(n_row, n_testing, row_start(row) + i)
            plotter(x_pred_i_i[i, ..., 0])
            plt.title('{} to {}'.format(digit_in, digit_in))
    
            row = 2
            plt.subplot(n_row, n_testing, row_start(row) + i)
            plotter(x_pred_i_j_i[i, ..., 0])
            plt.title('{} to {} to {}'.format(digit_in, digit_out, digit_in))
    
            row = 3
            plt.subplot(n_row, n_testing, row_start(row) + i)
            plotter(x_pred_i_j[i, ..., 0])
            plt.title('{} to {}'.format(digit_in, digit_out))

        plt.show()
        
    def testing_random(i_in = None, j_out = None):
        i_in_out = np.random.randint(0, n_digits, size=(2,))
        if i_in is None:
            digit_in = i_in_out[0]
        else:
            digit_in = i_in
            
        if j_out is None:
            digit_out = i_in_out[1]
        else:
            digit_out = j_out

        testing(digit_in, digit_out)
        
        

    def testing_disc():
        folder = '/home/lameeus/data/mnist/'
        
        x_gen = np.load(folder + 'x_gen.npy')
        # x_train2 = np.concatenate([x_train, x_gen], axis = 0)
        
        idx = np.arange(len(x_train))
        np.random.shuffle(idx)
        idx_gen = np.arange(len(x_gen))
        np.random.shuffle(idx_gen)
        
        x_train_sub = x_train[idx[0:10]]
        x_gen_sub = x_gen[idx_gen[0:10]]
        y_train_sub = model_d_im_code.predict(x_train_sub)
        y_gen_sub = model_d_im_code.predict(x_gen_sub)
        
        for i in range(8):
            plt.subplot(4, 4, i + 1)
            plt.imshow(x_train_sub[i, :, :, 0])
            argmax = np.argmax(y_train_sub[i, 0, 0, :])
            plt.title('{} : {}%'.format(argmax, 100 * y_train_sub[i, 0, 0, argmax]))
            plt.subplot(4, 4, i + 9)
            plt.imshow(x_gen_sub[i, :, :, 0])
            argmax = np.argmax(y_gen_sub[i, 0, 0, :])
            plt.title('{} : {}%'.format(argmax, 100 * y_gen_sub[i, 0, 0, argmax]))
        plt.show()

    def plot_super(i = None):
        if i is None:
            digit_in = np.random.randint(0, n_digits, size=(1,))[0]
        else:
            digit_in = i
        
        n_testing = 10
        
        x_orig_i = x_i_test[digit_in][:n_testing, ...]

        y_pred = dict_model_super[(digit_in)].predict(x_orig_i)
        
        print(len(y_pred))
        
        n_rows = 9
        
        for i in range(n_testing):
            plt.subplot(n_rows, n_testing, 1 + i)
            plotter(x_orig_i[i, ..., 0])
            plt.title('input'.format())
            
            i_row = 1
            plt.subplot(n_rows, n_testing, i_row * n_testing + 1 + i)
            plotter(y_pred[0][i, ..., 0])
            plt.title('dif auto'.format())

            i_row = 2
            plt.subplot(n_rows, n_testing, i_row * n_testing + 1 + i)
            plotter(x_orig_i[i, ..., 0] + y_pred[0][i, ..., 0])
            plt.title('auto C = {}'.format(y_pred[1][i, 0, 0, 0]))
            
            i_row = 3
            plt.subplot(n_rows, n_testing, i_row * n_testing + 1 + i)
            plotter(y_pred[2][i, ..., 0])
            plt.title('auto iji, C = {}'.format(y_pred[4][i, 0, 0, 0]))
            i_row = 4
            plt.subplot(n_rows, n_testing, i_row * n_testing + 1 + i)
            plotter(y_pred[3][i, ..., 0])
            plt.title('auto ijj, C = {}'.format(y_pred[4][i, 0, 0, 0]))
            
            i_row = 5
            plt.subplot(n_rows, n_testing, i_row * n_testing + 1 + i)
            plotter(y_pred[5][i, ..., 0])
            plt.title('auto iji, C = {}'.format(y_pred[7][i, 0, 0, 0]))
            i_row = 6
            plt.subplot(n_rows, n_testing, i_row * n_testing + 1 + i)
            plotter(y_pred[6][i, ..., 0])
            plt.title('auto ijj, C = {}'.format(y_pred[7][i, 0, 0, 0]))

            i_row = 7
            plt.subplot(n_rows, n_testing, i_row * n_testing + 1 + i)
            plotter(y_pred[8][i, ..., 0])
            plt.title('auto iji, C = {}'.format(y_pred[10][i, 0, 0, 0]))
            i_row = 8
            plt.subplot(n_rows, n_testing, i_row * n_testing + 1 + i)
            plotter(y_pred[9][i, ..., 0])
            plt.title('auto ijj, C = {}'.format(y_pred[10][i, 0, 0, 0]))
            
        plt.show()

        # x_pred_i_j = dict_model_g[(digit_in, digit_out)].predict(x_orig_i)
        # x_pred_i_i = dict_model_g[(digit_in, digit_in)].predict(x_orig_i)
        # x_pred_i_j_i = dict_model_g_trans[(digit_in, digit_out)].predict(x_orig_i)
        #
        # def row_start(row):
        #     return n_testing * row + 1
        #

        #
        # n_row = 4
        # for i in range(n_testing):
        #     row = 0
        #     plt.subplot(n_row, n_testing, row_start(row) + i)
        #     plotter(x_orig_i[i, ..., 0])
        #     plt.title('{} orig'.format(digit_in))
        #
        #     row = 1
        #     plt.subplot(n_row, n_testing, row_start(row) + i)
        #     plotter(x_pred_i_i[i, ..., 0])
        #     plt.title('{} to {}'.format(digit_in, digit_in))
        #
        #     row = 2
        #     plt.subplot(n_row, n_testing, row_start(row) + i)
        #     plotter(x_pred_i_j_i[i, ..., 0])
        #     plt.title('{} to {} to {}'.format(digit_in, digit_out, digit_in))
        #
        #     row = 3
        #     plt.subplot(n_row, n_testing, row_start(row) + i)
        #     plotter(x_pred_i_j[i, ..., 0])
        #     plt.title('{} to {}'.format(digit_in, digit_out))
        #
        # plt.show()
        
        
        
    """ Training """
    n_epochs = 100
    
    x_0_len = len(x_0)
    x_1_len = len(x_1)
    
    y_1_0 = np.zeros((x_1_len, 1, 1, 1))
    y_0 = np.ones((x_0_len, 1, 1, 1))
    y_0_1 = np.zeros((x_0_len, 1, 1, 1))
    y_1 = np.ones((x_1_len, 1, 1, 1))
    y_1_0_fool = np.ones((x_1_len, 1, 1, 1))
    y_0_1_fool = np.ones((x_0_len, 1, 1, 1))

    def generate_conv():
        x_1_0 = dict_model_g[(1, 0)].predict(x_1)
        x_0_1 = dict_model_g[(0, 1)].predict(x_0)
        return x_0_1, x_1_0

    def generate_like(i = None):
        # TODO
        if i:
            x_i_like = 0
            y_i_like = 0
            for j in range(n_digits):
                if i != j:
                    x_i_j = generate_conv(i)

                    x_i_like = np.concatenate()
            
            return x_i_like, y_i_like
        else:
        
            x_0_1, x_1_0 = generate_conv()
            
            x_0_like = np.concatenate([x_0, x_1_0], axis=0)
            y_0_like = np.concatenate([y_0, y_1_0], axis=0)
            x_1_like = np.concatenate([x_1, x_0_1], axis=0)
            y_1_like = np.concatenate([y_1, y_0_1], axis=0)
            
            return x_0_like, y_0_like, x_1_like, y_1_like
        
    def train_discr_all():
        """ regular discriminator """
        model_d_im_code.fit(x_train, y_train_cat,
                            epochs=1,
                            batch_size=32,
                            shuffle=True,
                            verbose=verbose
                            )
        
    def train_discr_random():
        """ Train the discriminator randomly """
        
        i = np.random.randint(0, n_digits, size=(1,))

        x_i_like, y_i_like = generate_like(i)

        logi = model_d_im_p[i].fit(x_i_like, y_i_like,
                                  epochs=1,
                                  batch_size=32,
                                  shuffle=True,
                                  verbose=verbose
                                  )

        eva_i = log2loss(logi)
        mean = np.mean([eva_i])
        losses['discr'].append(mean)
        
    def xy_gen():
        n_partly = None
    
        x_gen = np.zeros(shape=(0, 28, 28, 1))
        x_gen_auto = np.zeros(shape=(0, 28, 28, 1))
        y_gen_auto = np.zeros(shape=(0, 1, 1, 11))
        
        for i in range(n_digits):
            print('{}/{}'.format(i, n_digits))

            x_i_picked = x_i[i][:n_partly, ...]
            n_i = len(x_i_picked)
            for j in range(n_digits):
                if i != j:
                    x_ij = dict_model_g_trans[(i, j)].predict(x_i_picked)
                    print(np.shape(x_ij))
                    x_gen = np.concatenate([x_gen, x_ij], axis=0)
                if i == j:
                    x_ii = dict_model_g_trans[(i, i)].predict(x_i_picked)
                    x_gen_auto = np.concatenate([x_gen_auto, x_ii], axis=0)
                    y_ii = np.zeros(shape=(n_i , 1, 1, 11))
                    y_ii[:, :, :, i] = 1
                    y_gen_auto = np.concatenate([y_gen_auto, y_ii], axis=0)
    
        y_gen_cat = np.zeros(shape=(len(x_gen), 1, 1, 11))
        y_gen_cat[:, :, :, 10] = 1

        folder = '/home/lameeus/data/mnist/'
        
        np.save(folder +'x_gen.npy', x_gen)
        np.save(folder + 'y_gen.npy', y_gen_cat)
        
        np.save(folder + 'x_gen_auto.npy', x_gen_auto)
        np.save(folder + 'y_gen_auto.npy', y_gen_auto)
        
    def train_discr_all2():
        """ also takes in to account the generated
        """

        # xy_gen()

        folder = '/home/lameeus/data/mnist/'

        x_gen = np.load(folder + 'x_gen.npy')
        y_gen_cat = np.load(folder + 'y_gen.npy')
        
        x_gen_auto = np.load(folder + 'x_gen_auto.npy')
        y_gen_auto = np.load(folder + 'y_gen_auto.npy')

        # x_train2 = np.concatenate([x_train, x_gen], axis = 0)
        # y_train2 = np.concatenate([y_train_cat, y_gen_cat], axis = 0)
        x_train2 = np.concatenate([x_gen_auto, x_gen], axis=0)
        y_train2 = np.concatenate([y_gen_auto, y_gen_cat], axis=0)
        
        model_d_im_code.fit(x_train2, y_train2,
                            epochs=1,
                            batch_size=32,
                            shuffle=True,
                            verbose=1
                            )
        
        

    def log2loss(log):
        return log.history["loss"]
        
        
    def train_fooler_random():
        raise NotImplementedError
        
        
    def train_auto(i, add_log = True):
    
        # logii = dict_model_g[(i, i)].fit(x_i[i], x_i[i],
        logii = dict_model_auto[(i)].fit(x_i[i], x_i[i],
                                         epochs=5,
                                         batch_size=32,
                                         shuffle=True,
                                         validation_data=(x_i_test[i], x_i_test[i]),
                                         # callbacks = callbacks_list,
                                         verbose=verbose
                                         )
        
        if add_log:
            losses['auto'].append(log2loss(logii)[0])
        
        return logii
        
    def train_all_auto():
        eva_all = 0
        for i in range(n_digits):
            logii = train_auto(i, add_log = False)

            eva_all += log2loss(logii)[0]
        mean = eva_all/n_digits
        losses['auto'].append(mean)
        
    def train_auto_random():
        """ the same as train auto, but a random auto-encoder combination is chosen"""
        
        i_in_out = np.random.randint(0, n_digits, size=(2,))
        
        i = i_in_out[0]
        j = i_in_out[1]
        
        # auto-encoder
        dict_model_g[(i, i)].fit(x_i[i], x_i[i],
                                         epochs=1,
                                         batch_size=32,
                                         shuffle=True,
                                         # validation_data=(x_test, x_test),
                                         # callbacks = callbacks_list,
                                         verbose=verbose
                                         )
        
        # auto-encoder
        dict_model_g[(j, j)].fit(x_i[j], x_i[j],
                                         epochs=1,
                                         batch_size=32,
                                         shuffle=True,
                                         # validation_data=(x_test, x_test),
                                         # callbacks = callbacks_list,
                                         verbose=verbose
                                         )

       
        # through 'translation'
        logij = dict_model_g_trans[(i, j)].fit(x_i[i], x_i[i],
                                         epochs=1,
                                         batch_size=32,
                                         shuffle=True,
                                         # validation_data=(x_test, x_test),
                                         # callbacks = callbacks_list,
                                         verbose=verbose
                                         )
            
        eva_ij = log2loss(logij)
        mean = np.mean([eva_ij])
        losses['auto'].append(mean)
    

    def eva_val():
        x_0_1, x_1_0 = generate_conv()
        
        eva_0 = dict_model_d_i[0].evaluate(x_0, y_0, verbose=0)
        eva_1_0 = dict_model_d_i[0].evaluate(x_1_0, y_1_0, verbose=0)
        eva_1  = dict_model_d_i[0].evaluate(x_1, y_1, verbose=0)
        eva_0_1 = dict_model_d_i[0].evaluate(x_0_1, y_0_1, verbose=0)
        
        return eva_0, eva_1_0, eva_1, eva_0_1
        

    def eva():
        eva_0, eva_1_0, eva_1, eva_0_1 = eva_val()
        print('0: {}\n 1->0: {}\n 1: {}\n 0->1: {}'.format(eva_0[1], eva_1_0[1], eva_1[1], eva_0_1[1]))
    
    # eva()
    
    def eva_discr():
        # a = model_d_im_code.evaluate(x_train, y_train_cat, verbose = 0)
        # print(a)
        a = model_d_im_code.evaluate(x_test, y_test_cat, verbose = 0)
        print("discriminator: {}".format(a))
        # 1/0

    losses = {"discr": [], "gen": [], "auto": [], "super" : []}
    
    def train_super(i):
        x_i_select = x_i[i]
        
        y_super1 = np.zeros(shape=np.shape(x_i[i]))
        y_super2 = np.zeros(shape=(len(x_i[i]), 1, 1, 1))
        y_super3 = np.zeros(shape= (len(x_i[i]), shape_code[0],shape_code[1], shape_code[2] ) )     # compare code
        
        # y_super = [y_super1, y_super2] + [y_super1, y_super1, y_super2] * (n_digits - 1)
        # TODO super_model2
        # y_super = [x_i_select, y_super2, y_super3] + [x_i_select, y_super1, y_super2] * (n_digits-1)
        y_super = [x_i_select, y_super2, y_super3] + [x_i_select, y_super2] * (n_digits-1)

        
        logi = dict_model_super[i].fit(x_i_select, y_super,
                                epochs=10,
                                batch_size=32,
                                shuffle=True,
                                verbose=verbose,
                                callbacks=[TensorBoard(log_dir='/ipi/private/lameeus/data/mnist/log')]
                                )

        losses['super'].append(log2loss(logi))
        

    def train_super_random():
        i = np.random.randint(0, n_digits, size=(1,))[0]
        train_super(i)

    def train_super_all():
        for i in range(10):
            train_super(i)

    def train_super_deluxe():
        
        n_super_deluxe = min([len(x_i[i]) for i in range(n_digits)])
    
        x_i_select = [x_i[i][:n_super_deluxe, ...] for i in range(n_digits)]
        
        y_super1 = np.zeros(shape=np.shape(x_i_select[0]))
        y_super2 = np.zeros(shape=(n_super_deluxe, 1, 1, 1))
        y_super3 = np.zeros(shape= (n_super_deluxe, shape_code[0], shape_code[1], shape_code[2] ) )     # compare code

        y_super = []
        for i in range(n_digits):
            y_super_i = [x_i_select[i]] +  [x_i_select[i] , y_super3, y_super2] * (n_digits-1)
            y_super.extend(y_super_i)
        
        logi = model_super_deluxe.fit(x_i_select, y_super,
                                epochs=10,
                                batch_size=32,
                                shuffle=True,
                                verbose=verbose,
                                callbacks=[TensorBoard(log_dir='/ipi/private/lameeus/data/mnist/log')]
                                )

        losses['super'].append(log2loss(logi))
        

    verbose = 2
    def train_loop():
        for i_epoch in range(n_epochs):
            print('{}/{}'.format(i_epoch, n_epochs))
            
            """ adjust the discriminator """
            # train_discr_all()
            # train_discr_all2()
        
            """ train the auto-encoder """
            # train_auto()
            # train_auto_random()
            
            # train_auto(1)
            # train_auto(6)
            # train_auto(2)
            # train_auto(4)
            # train_auto(5)
        
            """ train everything at same time """
            # time_func( train_super_random )

            # for i in range(10):
            #     train_super(i)
            
            train_super_deluxe()

            # train_all_auto()
            
            # train_super(5)
            # train_super(4)
            # train_super(0)
            # train_super(6)
            
            if 0:
                """ out-dated """
                
                """ let the network fool the discriminator """
                # train_fooler()
                # train_fooler_random()
        
            # eva()
            # eva_discr()
        
            save_submodels()

    if 0:
        xy_gen()
        train_discr_all2()
        save_submodels()

    # train_super_all()
    
    # train_loop()
    
    # eva()
    
    def plot_training():
        plt.plot(losses['auto'], label='auto-encoder loss')
        plt.plot(losses['discr'], label='discriminator loss')
        plt.plot(losses['gen'], label='generator loss')
        plt.plot(losses['super'], label='super loss')
        plt.legend()
        plt.show()
    
    plot_training()
 
    # testing_random(i_in=0, j_out=None)

    # testing_disc()
    
    # testing_random(i_in=4, j_out=None)

    for i in range(10):
        # plot_super()
        testing_random(None, None)
        # testing_random(i_in = None, j_out = 6)
        # testing_random(i_in = 6, j_out = None)
    
    
if __name__ == '__main__':
    main()