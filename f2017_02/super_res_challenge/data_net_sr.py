import pickle
import numpy as np
import matplotlib.pyplot as plt

# 3th party
import data_net
import generate_h_images

# 778 is image of asian market
def data_test(im_index = 778):
    
    width = 32
    
    refdir = '//ipi/scratch/hluong/NTIRE17/DIV2K_train_HR'  # 5 MB per image, total of 4GB
    folder = '/scratch/lameeus/NTIRE17/DIV2K_train_HR_bicubic/X2-interpolation-bicubic'
    
    # # test other inputs
    # folder = '/scratch/lameeus/NTIRE17/DIV2K_train_HR_bicubic/X2-Quasar-shearlet'
    folder = '/scratch/lameeus/NTIRE17/DIV2K_train_HR_bicubic/X2-interpolation-bicubic'
    
    im_ref_name = refdir + '/' + "%04d" % im_index + '.png'
    
    im_inp_float = generate_h_images.return_image(im_index, folder=folder)
    im_ref_float = generate_h_images.image_float(im_ref_name)
    
    data = data_net.Data(im_inp_float, im_ref_float, big_patches = width, ext = 7)
    return data

def data_train(width):
    im_index = 700    #0763 Butterfly
    # im_index = 701  # 0763
    im_index = 763
    
    refdir = '//ipi/scratch/hluong/NTIRE17/DIV2K_train_HR'  # 5 MB per image, total of 4GB
    # Bicubic folder
    folder = '/scratch/lameeus/NTIRE17/DIV2K_train_HR_bicubic/X2-interpolation-bicubic'
    
    im_ref_name = refdir + '/' + "%04d" % im_index + '.png'
    
    im_inp_float = generate_h_images.return_image(im_index, folder = folder)
    im_ref_float = generate_h_images.image_float(im_ref_name)
    
    data = data_net.Data(im_inp_float, im_ref_float, big_patches = width)
    return data

# todo patches_input_shuffled of multiple images
def gen_patches_input_shuffled(bool_new=True, part = 0):
    main_dir = '/scratch/lameeus/NTIRE17/lameeus/'

    if part == 0:
        patches_dir = main_dir + 'patches_46_32_x2/'
    if part == 1:
        patches_dir = main_dir + 'patches_46_32_x2_shearlet/'

    save_name1 = patches_dir + 'patch_in.p'
    save_name3 = patches_dir + 'patch_in_that.p'
    save_name2 = patches_dir + 'patch_out.p'
    
    # the input
    folder = '/scratch/lameeus/NTIRE17/DIV2K_train_HR_bicubic/X2-Quasar-shearlet'
    # the output
    refdir = '//ipi/scratch/hluong/NTIRE17/DIV2K_train_HR'  # 5 MB per image, total of 4GB

    width_in = 46
    width_out = 32

    if bool_new:
    
        im_amount = 800
        # TODO REMOVE
        # im_amount = 2
    
        patches_input_shuffled = np.empty(shape=(0, width_in, width_in, 1))
        patches_input_tophat = np.empty(shape=(0, width_in, width_in, 1))
        patches_output_shuffled = np.empty(shape=(0, width_out, width_out, 1))
    
        for im_index in range(1, im_amount + 1):
            im_inp_float_i = generate_h_images.return_image(im_index, folder=folder)
            im_ref_name = refdir + '/' + "%04d" % im_index + '.png'
            im_ref_float_i = generate_h_images.image_float(im_ref_name)
            data_i = data_net.Data(im_inp_float_i, im_ref_float_i, big_patches=width_out, ext=7)
            # (patches_input_shuffled_i, patches_output_shuffled_i) = data_i.images2patches()

            patches_input_shuffled_i = data_i.in_patches()
            patches_input_that_i = data_i.in_patches_gausshat()
            patches_output_shuffled_i = data_i.out_patches()
        
            subset_am = 200
        
            patches_input_shuffled = np.concatenate((patches_input_shuffled,
                                                     patches_input_shuffled_i[0:subset_am, ...]), axis=0)
            patches_input_tophat = np.concatenate((patches_input_tophat,
                                                 patches_input_that_i[0:subset_am, ...]), axis=0)
            
            patches_output_shuffled = np.concatenate((patches_output_shuffled,
                                                      patches_output_shuffled_i[0:subset_am, ...]), axis=0)
        
            print("at image {}".format(im_index))
            # print(np.shape(patches_input_shuffled_i))
    
        print(np.shape(patches_input_shuffled))
        print(np.shape(patches_output_shuffled))
    
        pickle.dump(patches_input_shuffled, open(save_name1, "wb"))
        pickle.dump(patches_input_tophat, open(save_name3, "wb"))
        pickle.dump(patches_output_shuffled, open(save_name2, "wb"))

    else:
    
        patches_input_shuffled = pickle.load(open(save_name1, "rb"))
        patches_input_tophat = pickle.load(open(save_name3, "rb"))
        patches_output_shuffled = pickle.load(open(save_name2, "rb"))

    return (patches_input_shuffled, patches_input_tophat, patches_output_shuffled)

def tophatblur(im, sigma = 7):
    from scipy.ndimage.filters import gaussian_filter
    
    def blurrer(im, sigma = 0):

        shape = np.shape(im)
        
        im_blurred = np.empty(shape)
        for d_i in range(shape[-1]):
            im_blurred[..., d_i] = gaussian_filter(im[..., d_i], sigma=sigma)
        
        return im_blurred
    
    im_blurred = blurrer(im, sigma=sigma)
    return (im - im_blurred)
    
# To tophat tf etc
def main():
    data = data_test()
    im_in = data.im_in
    
    plt.figure()
    plt.subplot('311')
    plt.imshow(im_in)
    plt.title('im_in')
    
    R = im_in[..., 0]
    G = im_in[..., 1]
    B = im_in[..., 2]



    plt.subplot('312')

    plt.title('im_blurred')

    plt.subplot('313')
    im_dif = tophatblur(im_in, sigma=7)
    plt.imshow(im_dif)
    plt.title('im_dif')
        
    plt.figure()
    
    im_in_flat = np.reshape(im_in, newshape=(-1,))
    # im_blurred_flat = np.reshape(im_blurred, newshape=(-1,))
    im_dif_flat = np.reshape(im_dif, newshape=(-1,))

    plt.subplot('311')
    n, bins, patches = plt.hist(im_in_flat, 256, normed=1, facecolor='green', alpha=0.75)
    plt.title('im_in')
    plt.subplot('312')
    # n, bins, patches = plt.hist(im_blurred_flat, 256, normed=1, facecolor='green', alpha=0.75)
    plt.title('im_blurred')
    plt.subplot('313')
    n, bins, patches = plt.hist(im_dif_flat, 256, normed=1, facecolor='green', alpha=0.75)
    plt.title('im_dif')
    plt.show()

if __name__ == '__main__':
    main()
    