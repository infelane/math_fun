import scipy.misc

import data_net_sr
from main_sr import *
from link_to_soliton.paint_tools import image_tools


# Generate each output image
def main():
    range_im = 800  # 800
    range_im = 10

    bool_show = False
    
    # Get Network
    layers = config.layer_config3()
    
    network_group = network.NetworkGroup(layers=layers, bool_residue= True)
    
    savedir = "/scratch/lameeus/NTIRE17/lameeus/x2_cnn_bicubic"
 
    flag = config.FLAGS1()
  
    network_group.load_params(flag.checkpoint_dir)
    
    for index_im in range(1, range_im + 1):

        data = data_net_sr.data_test(im_index=index_im)
        
        # generate H image
        im_lam = net2h_image(networkGroup=network_group,
                             data=data
                             )

        if bool_show:
            plt.imshow(im_lam)
            plt.show()

        savename = savedir + '/' + "%04d" % int(index_im) + 'x2.png'
        print(savename)

        # save H image
        scipy.misc.toimage(im_lam, cmin=0.0, cmax=1.0).save(savename)
        

        
    network_group.close_session()


def return_image(name_im, folder=None):
    x = 2
    
    if folder is None:
        dirname = '//ipi/scratch/hluong/NTIRE17/DIV2K_train_LR_bicubic/X' + str(x)
    else:
        dirname = folder
    
    i = int(name_im)
    
    im_inp_name = dirname + '/' + "%04d" % i + 'x' + str(x) + '.png'
    # return image_float(im_inp_name)
    return image_tools.path2im(im_inp_name)[..., 0:3]


def image_float(name):
    im_inp_float = np.asarray(Image.open(name)) / 255
    return im_inp_float[..., 0:3]   # remove the transparancy if needed


def net2h_image(networkGroup=None, data=None, tophat_bool = True):

    # Split up the output in smaller patches
    in_patches = data.in_patches()
    batch_size = 100
    batch_amount = int(np.ceil(np.shape(in_patches)[0]/batch_size))
    out = None
    
    for batch_i in range(batch_amount):
        feed_dict = {networkGroup.x: data.in_patches()[batch_i*batch_size:(batch_i+1)*batch_size]}
        if tophat_bool:
            feed_dict.update({networkGroup.x_tophat: data.in_patches_gausshat()[batch_i*batch_size:(batch_i+1)*batch_size]})
        
        out_i = networkGroup.get_output(feed_dict=feed_dict)
        if out is None:
            out = out_i
        else:
            out = np.append(out, out_i, 0)
                        
    im_lam = data.patches2images(out)

    def build_dict(data_placeholder):
        feed_dict = {networkGroup.x: data_placeholder.x}
        if tophat_bool:
            feed_dict.update({networkGroup.x_tophat: data_placeholder.x_tophat})
        return feed_dict
        
    data_placeholder = data.right_patches()
    feed_dict = build_dict(data_placeholder)
    out = networkGroup.get_output(feed_dict=feed_dict)

    im_right = data.right_patches2images(out)
    
    width = data.width

    data_placeholder = data.botright_patch()
    feed_dict = build_dict(data_placeholder)
    out = networkGroup.get_output(feed_dict=feed_dict)
    im_lam[-width:, -width:, :] = data.botright_patch2image(out)

    shape_im = data.shape
    for h_i in range(int(shape_im[0]/width)):
        im_lam[h_i*width: (h_i+1)*width, -width:, :] = im_right[h_i]

    data_placeholder = data.bot_patches()
    feed_dict = build_dict(data_placeholder)
    out = networkGroup.get_output(feed_dict=feed_dict)

    im_bot = data.bot_patches2images(out)
        
    shape_im = data.shape
    for w_i in range(int(shape_im[1]/width)):
        im_lam[-width:, w_i*width: (w_i+1)*width, :] = im_bot[w_i]
        
    # plt.imshow(foo[0 : 1344, :, :])
    # plt.show()
    
    # shape = np.shape(im_inp_float)
    #
    # h = shape[0]
    # w = shape[1]
    #
    # im_lam = np.zeros(shape=(2 * h, 2 * w, 3))
    #
    # for h_i in range(h):
    #     inp_i = np.zeros((w, 3, 3, 3))
    #     for w_i in range(w):
    #         inp_i[w_i, ...] = image_extended.get_segm(h_i, w_i)
    #
    #     feed_dict = {placeholders.x: inp_i}
    #     out_i = networkGroup.get_output(placeholders, feed_dict=feed_dict)
    #
    #     for w_i in range(w):
    #         im_lam[2 * h_i:2 * h_i + 2, 2 * w_i:2 * w_i + 2, :] = out_i[w_i, ...]
    #
    # # TODO in_image => placeholders.x: input, get output, output to output_im
    #
    # im_lam[im_lam > 1.0] = 1.0
    # im_lam[im_lam < 0.0] = 0.0
    
    return im_lam


if __name__ == '__main__':
    main()
