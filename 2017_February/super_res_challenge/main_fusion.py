"""
Fuses multiple preprocessors
"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# My Modules
import main_sr
import network
import config
import data_net
import generate_h_images


class Settings():
    folder_wb = '/ipi/private/lameeus/data/super_res/net_fusion/wb/'
    load_prev = True
    train = False
    lr = 1.0e-7
    epochs = 10000
    batch_size = 100

def main():
    # Settings:
    settings = Settings()
    
    # Data
    folder_name_pre = '/scratch/lameeus/NTIRE17/DIV2K_train_HR_bicubic/'
    folder_name_bicubic = folder_name_pre + 'X2-interpolation-bicubic/'
    folder_name_tv = folder_name_pre + 'X2-Matlab-tv/'
    folder_name_dtcwt = folder_name_pre + 'X2-Quasar-dtcwt/'
    folder_name_shearlet = folder_name_pre + 'X2-Quasar-shearlet/'
    folder_name_wavelet = folder_name_pre + 'X2-Quasar-wavelet/'
    folder_name_srcnn = folder_name_pre + 'X2-SRCNN/'
        
    folder_name_cnn = '/scratch/lameeus/NTIRE17/lameeus/'
    folder_name_cnn_bicubic = folder_name_cnn + 'x2_cnn_bicubic/'
    folder_name_cnn_tv = folder_name_cnn + 'x2_cnn_matlab-tv/'
    folder_name_cnn_dtcwt = folder_name_cnn + 'x2_cnn_dtcwt/'
    folder_name_cnn_shearlet = folder_name_cnn + 'x2_cnn_shearlet/'
    folder_name_cnn_wavelet = folder_name_cnn + 'x2_cnn_wavelet/'
    folder_name_cnn_srcnn = folder_name_cnn + 'x2_cnn_srcnn/'
    
    
    folder_ground_truth = '//ipi/scratch/hluong/NTIRE17/DIV2K_train_HR'

    ### Input arrays
    # order of inputs
    names = ["bicubic", "tv", "dtcwt", "shearlet", "wavelet", "srcnn",
             "bicubic_cnn", "tv_cnn", "dtcwt_cnn", "shearlet_cnn", "wavelet_cnn", "srcnn_cnn"]
    folder_inputs = [folder_name_bicubic, folder_name_tv, folder_name_dtcwt,
                     folder_name_shearlet, folder_name_wavelet, folder_name_srcnn,
                     folder_name_cnn_bicubic, folder_name_cnn_tv, folder_name_cnn_dtcwt,
                     folder_name_cnn_shearlet, folder_name_cnn_wavelet, folder_name_cnn_srcnn]
    
    n_input = len(names)
    n_train = 10

    im_in = []  # list of all the inputs of each image
    im_out = [] # list of the output of each image
    
    for im_nr_i in range(1, n_train+1):
    # for im_nr_i in range(6, 6 + n_train):
        nr_name = '{0:04d}'.format(im_nr_i)

        im_in_i = []

        for n_input_i in range(n_input):
                        
            name_in_i = folder_inputs[n_input_i] + nr_name + 'x2.png'
            im_in_i.append(open_image(name_in_i))

        name_out = folder_ground_truth + '/' + nr_name + '.png'

        im_in.append(im_in_i)
        im_out.append(open_image(name_out))
        
        
    # im_nr = 3

    # grey_input = np.stack([np.mean(im_in_i, axis = -1) for im_in_i in im_in], axis=2)
    # shape = np.shape(im_out)
    # grey_output = np.reshape(np.mean(im_out, axis=-1), newshape=(shape[0], shape[1], 1))
    
    # print(main_sr.performance(grey1to3(grey_input[..., 0]), grey1to3(grey_output[..., 0])))
    
    layers = config.layer_config_fus0()
    network_group = network.NetworkGroup(layers=layers)
    network_group.set_lr(lr = settings.lr)
    network_group.set_train()

    # data = [flatten(grey_input), flatten(grey_output)]

    patch_width = layers.layer_size[-1][0]
    
    # data = data_net.Data(grey_input, grey_output, bool_tophat = False, colors_sep=False,
    #                      big_patches=patch_width, ext = 2)
    #
    # data = data_net.DataSet2(images = data.patches_input, labels = data.patches_output)
    
    if settings.load_prev:
        network_group.load_params(settings.folder_wb)
       
    else:
        network_group.load_init()

    if settings.train:
        # Training
        images = []
        labels = []
        for im_i in range(n_train):
            
            # for in_i in range(n_input):
            #     print(len(im_in))
            #     print(np.shape(im_in))

            # Only the green input is taken, is better because of the chromatic aberation
            green_input = np.stack([im_in_i[..., 1] for im_in_i in im_in[im_i]], axis=2)
            
            shape = np.shape(im_out[im_i])
            green_output = np.reshape(im_out[im_i][..., 1], newshape=(shape[0], shape[1], 1))

            data = data_net.Data(green_input, green_output, bool_tophat=False, colors_sep=False,
                                 big_patches=patch_width, ext=2)

            images.append(data.patches_input)
            labels.append(data.patches_output)

        images = np.concatenate(images, axis = 0)
        labels = np.concatenate(labels, axis = 0)
        data_train =  data_net.DataSet2(images = images, labels = labels)
        
        train_loop(network_group, data_train, settings)

    im_gen = []
    for im_index in range(n_train):
        im_gen_i = show_results1(network_group, im_in[im_index], im_out[im_index])
        im_gen.append(im_gen_i.astype(np.float64))
        
    show_results2(im_in, im_out, im_gen, names)
    
    # release resources
    network_group.close_session()
    
    
def show_results2(im_in, im_out, im_gen, names):
    
    perf_in = []
    
    n_im = len(im_in)
    n_im_in = len(im_in[0])

    mean = lambda l: sum(l) / len(l)
    for n_im_in_i in range(n_im_in):
    
        perf_in_i = []
        for n_im_i in range(n_im):
            psnr_i = main_sr.performance(im_in[n_im_i][n_im_in_i], im_out[n_im_i], ssim_bool = False)
            # only psnr
            perf_in_i.append(psnr_i[1])
        
        perf_in.append(mean(perf_in_i))

    psnr = []
    for n_im_i in range(n_im):
        psnr_i = main_sr.performance(im_gen[n_im_i], im_out[n_im_i], ssim_bool=False)[1]
        psnr.append(psnr_i)
    perf_gen = mean(psnr)
    
    print('Input PSNR:\n{}'.format(perf_in))
    print('Generated PSNR:\n{}'.format(perf_gen))
    
    plt.figure()

    from matplotlib.pyplot import cm
    color = cm.rainbow(np.linspace(0, 1, int(n_im_in/2)+1))

    for n_im_in_i in range(n_im_in):
        c = color[n_im_in_i%6]
        if n_im_in_i//6: # with preprocessing
            opts = '-'
        else:
            opts = '--'
        y = perf_in[n_im_in_i]
        plt.plot([0, 1], [y, y], opts, label = names[n_im_in_i], c = c )
    c = color[-1]
    y=perf_gen
    opts = ':'
    plt.plot([0, 1], [y, y], opts, label='generated', c=c)
    plt.legend()
    # plt.show()
    
    # output and difference
    plt.figure()
    plt.subplot('221')
    plt.imshow(im_gen[0])
    plt.title('generated')
    plt.subplot('222')
    plt.imshow(im_gen[0] - im_out[0] + 0.5)
    plt.title('difference')
    plt.subplot('223')
    plt.imshow(im_out[0])
    plt.title('ground truth')
    plt.show()
    
    
def show_results1(network_group, im_in, im_out):
    R = np.stack([im_in_i[..., 0] for im_in_i in im_in], axis=2)
    G = np.stack([im_in_i[..., 1] for im_in_i in im_in], axis=2)
    B = np.stack([im_in_i[..., 2] for im_in_i in im_in], axis=2)
    
    shape = np.shape(im_out)
    
    # data_R = flatten(R)
    # data_G = flatten(G)
    # data_B = flatten(B)
    
    def gen_data(data_i):
        patch_width = network_group.a_layers.layer_size[-1][0]
        
        data = data_net.Data(data_i, data_i[..., 0:1], bool_tophat=False, colors_sep=False,
                             big_patches=patch_width, ext=2)
           
        return data
        # data = data_net.DataSet2(images=data.patches_input, labels=data.patches_output)

    # data = [flatten(grey_input), flatten(grey_output)]

    out_R = generate_h_images.net2h_image(network_group, gen_data(R), tophat_bool = False)
    out_G = generate_h_images.net2h_image(network_group, gen_data(G), tophat_bool = False)
    out_B = generate_h_images.net2h_image(network_group, gen_data(B), tophat_bool = False)
    

    # out_R = network_group.get_output({network_group.x: data_R})
    # out_G = network_group.get_output({network_group.x: data_G})
    # out_B = network_group.get_output({network_group.x: data_B})
    
    # print(np.shape(out_B))
    #
    # data = np.concatenate([out_R, out_G, out_B], axis = 3)
    # h = shape[0]
    # w = shape[1]
    #
    # im_gen = deflatten(data, h, w)
    
    im_gen = np.concatenate([out_R, out_G, out_B], axis = 2)
    
    # adjusting outliers
    im_gen[im_gen > 1.0] = 1.0
    im_gen[im_gen < 0.0] = 0.0
    
    # plt.imshow(im_gen)
    # plt.show()
    
    return im_gen
    
    
def show_results0(im_in, im_out):
    plt.figure()
    plt.subplot('211')
    plt.imshow(im_in[0])

    plt.subplot('212')
    plt.imshow(im_out)

    print(main_sr.performance(im_in[0], im_out))

    plt.show()

def flatten(data):
    """ Convert image to an array of 1 by 1 inputs """
    shape = np.shape(data)
    if len(shape) == 2:
        return np.reshape(data, newshape=(shape[0]*shape[1], 1, 1, 1))
    if len(shape) == 3:
        return np.reshape(data, newshape=(shape[0]*shape[1], 1, 1, shape[2]))
    else:
        raise ValueError('Input is expected to be len(shape) 2 or 3')
    
def deflatten(data, h, w):
    return np.reshape(data, newshape=(h, w, 3))

def grey1to3(im):
    """ Converts a single channel grey to 3 channel"""
    return np.stack([im, im, im], axis = 2)
    

def train_loop(network_group, data, settings):
    
    batch_iters = data.num_examples//settings.batch_size
    batch_iters = 1000
    for step in range(settings.epochs):
        for _ in range(batch_iters):  # Go over all the mini batches
            
            batch_i = data.next_batch(settings.batch_size)
            
            feed_dict = {network_group.x: batch_i.x,
                         network_group.y: batch_i.y}

            network_group.train(feed_dict)

        batch_test = data.get_test_data()
        feed_dict = {network_group.x: batch_test.x,
                     network_group.y: batch_test.y}

        cost = network_group.cost(feed_dict)

        print('cost {}'.format(cost))
        
        network_group.plus_global_epoch()
        
        # save network
        network_group.save_params(settings.folder_wb)


def open_image(file_name):
    return np.asarray(Image.open(file_name))[..., 0:3]/255  # removes transparancy

if __name__ == '__main__':
    main()
    