# guess porosity of tablets

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

bit_max = 65536 - 1     # 16bit
bit_max = 256 - 1     # 16bit

h_top = 232  # Transition zone between top and middle pill
h_bot = 1121  # Transition zone between middle and bottom pill
r_pill = 877  # radius of the pill


def main():
    if 0:  # if you want to rerotate the images
        if 0:
            data_all = open_orig()
        else:  # Data from Jan, preprocessed holes
            data_all = open_jan()
        
        if 1:
            cross_section(data_all, norm_im=False)
            plot_inter(data_all, norm_im=False)
        
        data_rot = rotation(data_all)
        
        if 0:
            cross_section(data_rot, norm_im=False)
            plot_inter(data_rot, norm_im=False)
        
        save_interstate(data_rot, folder_name='/scratch/lameeus/data/pharmacy_tablets/rot_holes/')
    
    else:  # load rotated image
        # data_rot = open_rot(folder_name='/scratch/lameeus/data/pharmacy_tablets/rot_holes/')

        # cross_section(data_rot, norm_im=False)
        # plot_inter(data_rot, norm_im=False)

        # filter_cross_section(data_rot)
        
        scale_color()
        

def gen_3d_folder(n_begin, n_end, folder, ext = 'tif'):
    i_im = np.arange(n_begin, n_end + 1, dtype=int)
    
    def foo(index):
        im_name_i = folder + '3_{0:05d}.{1}'.format(index, ext)
        im = Image.open(im_name_i)
        imarray = np.array(im, dtype=np.uint16)
        return imarray
    
    data_all = np.stack([foo(index) for index in i_im], axis=0)
    
    return data_all

def open_orig():
    n_begin = 500  # 500 to 1700
    n_end = 1700
    folder = '/net/ipids/microscopy/CT/pharmacy_tablets/'
    lambda0 = lambda : gen_3d_folder(n_begin, n_end, folder)
    
    data_all = time_func(lambda0)  # 14s
    
    return data_all


def open_jan():
    n_begin = 500  # 500 to 1693
    n_end = 1693
    folder = '/net/ipids/microscopy/CT/temp/holes/'
    lambda0 = lambda: gen_3d_folder(n_begin, n_end, folder, 'png')
    
    data_all = time_func(lambda0)  # 14s
    
    return data_all

def open_rot(folder_name = '/scratch/lameeus/data/pharmacy_tablets/rotated/'):

    def gen_3d():
        def foo(index):
            im_name_i = folder_name + 'cut_{}.tif'.format(index)
            im = Image.open(im_name_i)
            imarray = np.array(im, dtype=np.uint16)
            return imarray
    
        nr_images = 1284 # 1291 max
    
        i_im = np.arange(0, nr_images +1)
        data_rot = np.stack([foo(index) for index in i_im], axis=0)
        return data_rot

    data_rot = time_func(gen_3d)
    return data_rot


def int16to8(map3d):

    map3d_new = map3d // 256
    map3d_new = map3d_new.astype(np.int8)
    return map3d_new

    # data_int8 = int16to8(data_all)
    
    # lambda0 = lambda : np.save(name_3d, data_all)
    # time_func(lambda0)  # 84 s
    # lambda0 = lambda : np.save(name_3d, data_int8)
    # time_func(lambda0)  # 42 s
    
    # np.save(name_3d, data_all)

    lambda0 = lambda: np.load('3dimage.npy')
    data_all = time_func(lambda0)
        
    
def scale_color():
    # 0: no preprocessing
    # 1: holes preprocessing
    bool_preproc = 1
    
    if bool_preproc:
        folder = '/scratch/lameeus/data/pharmacy_tablets/results_from_holes/'
    else:
        folder = '/scratch/lameeus/data/pharmacy_tablets/results/'
   
    im_name = folder + 'results.tif'
    # im_name = '/ipi/private/lameeus/private_Documents/python/2017_05/results.tif'
    
    im = Image.open(im_name)
    array = np.array(im)
    
    array = crop2(array, 100, 1)
    
    if 0:
        intens, bins = np.histogram(array, bins=256, range=[0, bit_max])
    
        bins_center = (bins[0:-1] + bins[1:]) / 2.0
    
        plt.plot(bins_center, intens)
        plt.show()
        
    else:
        val_max = 40000 # guessed based on histogram
        
    print(np.max(array))

    if bool_preproc:
        array_norm = array/255
    else:
        array_norm = 1 - array/val_max

    plt.imshow(array_norm, vmin = 0.044, vmax = 0.155, cmap = 'jet')
    if bool_preproc:
        plt.title('cross section after mathematical morphology')
    else:
        plt.title('cross section by averaging raw images')
    plt.colorbar()
    plt.show()
    

def time_func(func):
    start = time.time()
    a = func()
    end = time.time()
    print('elapsed time: {}'.format(end - start))
    return a


def plot_inter(map3d, norm_im = True):

    shape = np.shape(map3d)
    hor_index = (shape[2]//2)
    # cross_hor = []

    i_plot = 0
    
    d_step = hor_index//5
    
    for hor_index_i in range(hor_index - 4*d_step, hor_index + 4*d_step+1, d_step):
        cross_hor_i = map3d[:, :, hor_index_i]
        # cross_hor.append(cross_hor_i)

        i_plot += 1
        plt.subplot(3, 3, i_plot)
        if norm_im:
            norm_imshow(cross_hor_i)
        else:
            plt.imshow(cross_hor_i, vmin = 0)

    plt.show()
    
    
def save_interstate(map3d, folder_name= '/scratch/lameeus/data/pharmacy_tablets/'):
    
    shape = np.shape(map3d)
    d = shape[0]
    w = shape[2]
    h = shape[1]
    for i in range(d):
        name = folder_name + 'cut_{}.tif'.format(i)
        
        array_i = map3d[i, :, :]
        
        im = Image.frombytes('I;16', (w, h), array_i.tobytes())
        im.save(name)
        
        print('{} / {}'.format(i, d))
    
    
def filter_cross_section(map3d):
    
    if 0:   # TODO more educated guess on dust
        intens, bins = np.histogram(map3d[::2,::2,::2], bins = 256, range=[0, bit_max])
    
        bins_center = (bins[0:-1] + bins[1:])/2.0
    
        print(np.shape(bins_center))
        print(np.shape(intens))
    
        plt.plot(bins_center, intens)
        plt.show()
        
    else:
        c_mean_guess = 36990. # For original
        c_mean_guess = 255 # For holes (a higher value is air)
        
    # Settings
    step_size = 1
    r_range = np.arange(0., 901., step_size, dtype=float)   # 0 to 901, per 10
    h_range = np.arange(100, 1200, step_size, dtype=int)    # 100 to 1200 per 10
    std = 1.0     # 5.0

    # calculated values
    height_range = int(2*std)
    center = ((129.5 + 1883.)/2.0, (111.5 + 1862.0)/2.0)
    
    r_len = len(r_range)
    h_len = len(h_range)
    results = np.zeros((h_len, r_len))
    for i_r in range(r_len-1, -1, -1):
        r_3d = r_range[i_r]
        
        # calculated values, independent of height
        dist_inner = (r_3d - 2*std)/1.42 # TODO not used yet
        dist_outer = r_3d + 2.*std
        w_pixel = [int(np.floor(center[0] - dist_outer)), int(np.ceil(center[0] + dist_outer))]
        d_pixel = [int(np.floor(center[1] - dist_outer)), int(np.ceil(center[1] + dist_outer))]
        shape_filter = (1 + 2*height_range, w_pixel[1] - w_pixel[0], d_pixel[1] - d_pixel[0])

        """ The filter build part """
        filter_all = np.zeros(shape=shape_filter)
        dist_sq_all = np.zeros(shape=shape_filter)

        h_i_range = np.reshape(np.arange(shape_filter[0]), newshape=(-1, 1, 1))
        w_i_range = np.reshape(np.arange(shape_filter[1]), newshape=(1, -1, 1))
        d_i_range = np.reshape(np.arange(shape_filter[2]), newshape=(1, 1, -1))

        delta_w_range = w_i_range - center[0] + w_pixel[0]
        delta_d_range = d_i_range - center[1] + d_pixel[0]
        delta_h = h_i_range - height_range

        delta_w_sq = np.square(delta_w_range)
        delta_d_sq = np.square(delta_d_range)

        dist_sq_range_1 = np.square(r_3d - np.sqrt(delta_w_sq + delta_d_sq))
        dist_sq_range_2 = np.square(delta_h)
        
        dist_sq_all[:, :, :] += dist_sq_range_1 + dist_sq_range_2
    
        dist_norm = (dist_sq_all) / (2.0 * std ** 2)
        select = (dist_norm < 5)
        filter_all[select] = np.exp(-dist_norm[select])
        
        # normalization
        filter_all /= np.sum(filter_all)
        
        if 0:
            # Visualization of the kernel
            plot_inter(filter_all, norm_im=False)
        
        for i_h in range(h_len):
            h_3d = h_range[i_h]
            
            crop_3d = map3d[h_3d - height_range: h_3d + height_range + 1, w_pixel[0]: w_pixel[1],
                      d_pixel[0]: d_pixel[1]]

            value = np.sum(crop_3d * filter_all)

            results[i_h, i_r] = value
    
        print('progress: {} / {} goes slower towards the end'.format(i_r, r_len))
        
    # normalize
    results = results/float(c_mean_guess)
    
    results = results.astype(float)
    import scipy.misc
    scipy.misc.toimage(results, cmin=0.0, cmax=1.0).save('results.tif')

    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=0, vmax=1.0)
    im = cmap(norm(results))
    plt.imsave('colormap.png', im)
    
    # Crop the results
    results = crop2(results, h_range_0=h_range[0] ,step_size = step_size)
    
    vmax = 0.2
    
    norm = plt.Normalize(vmin=0, vmax=vmax)
    im = cmap(norm(results))
    plt.imsave('colormap_final.png', im)
    plt.imshow(results, vmin = 0.0, vmax = vmax, interpolation='nearest', cmap = 'jet')
    plt.colorbar()
    plt.show()
    
            # def build_filter(r):
    #     filter = np.zeros((2 * depth_range + 1, shape[1], shape[2]), dtype=float)
    #
    #     return filter
    #
    # filter = build_filter(r0)
    #
    # max_range = r0 + 2*std
    # min_range = (r0 - 2*std)/1.42
    # for h_i in range(2*depth_range + 1):
    #
    #     w_i_arange = np.arange(shape[1])
    #
    #     delta_w_ar = w_i_arange - center[0]
    #
    #
    #
    #
    #     # for w_i in range(shape[1]):
    #     for d_i in range(shape[2]):
    #     #         delta_w = w_i - center[0]
    #             delta_d = d_i - center[1]
    #     #
    #     #         if abs(delta_w) > max_range:
    #     #             continue
    #     #         if abs(delta_d) > max_range:
    #     #             continue
    #     #         if (abs(delta_w) < min_range) & (abs(delta_d) < min_range):
    #     #             continue
    #     #         # if :
    #     #         #     continue
    #     #
    #     #         dist_sq = (depth_range - h_i)**2 + np.abs(r0**2 - (delta_w)**2 - (delta_d)**2)
    #     #         filter[h_i, w_i, d_i] = np.exp(-(dist_sq)/(2.0*std**2))
    #
    #             dist_sq = (depth_range - h_i) ** 2 + np.abs(r0 ** 2 - (delta_w_ar) ** 2 - (delta_d) ** 2)
    #             filter[h_i, (abs(delta_w_ar) > max_range), d_i] = np.exp(-(dist_sq) / (2.0 * std ** 2))

    # plt.imshow(filter[depth_range, :, :])
    # plt.show()
    #
    # end = time.time()
    # print('elapsed time: {}'.format(end - start))
    #
    #
    #
    # start = time.time()
    #
    # filter_sum = np.sum(filter)
    # filter /= filter_sum
    #
    #
    #
    # concentration = np.zeros((10, 1500))
    # for h_i in range(20):
    #
    #     start_index = 200
    #     value = np.sum(filter * map3d[h_i + start_index- depth_range:h_i + start_index + depth_range+1, : ,:])
    #     concentration[0, h_i] = value
    #
    # end = time.time()
    # print('elapsed time: {}'.format(end - start))
    #
    # plt.imshow(concentration)
    # plt.show()

    
def crop(results, h_range_0 = 0, step_size = 1):
    # set everything to zero
    results[0: int((h_top -h_range_0) / step_size), :] = 0
    results[int((h_bot -h_range_0) / step_size) + 1:, :] = 0
    results[:, int(r_pill / step_size) + 1:] = 0
    
    return results

def crop2(results, h_range_0 = 0, step_size = 1):
    # removes borders

    results = results[int((h_top -h_range_0) / step_size): int((h_bot -h_range_0) / step_size) + 1,
              : int(r_pill / step_size) + 1]
    
    return results
  

def rotation(map3d):
    top_hor = np.array([[128, 159], [1875, 214]]) # width, height, width, height
    top_ver =np.array( [[89, 203], [1842, 171]])
    
    bot_hor =np.array( [[101, 1047], [1853, 1104]])
    bot_ver = np.array([[108, 1091], [1858, 1058]])
    
    # height
    top_center_h = (top_hor[0, 1] + top_hor[1, 1] + top_ver[0, 1] + top_ver[1, 1])/4.0
    bot_center_h = (bot_hor[0, 1] + bot_hor[1, 1] + bot_ver[0, 1] + bot_ver[1, 1])/4.0
    # width
    top_center_w = (top_hor[0, 0] + top_hor[1, 0])/2.0
    # depth
    top_center_d = (top_ver[0, 0] + top_ver[1, 0]) / 2.0
    bot_center_w = (bot_hor[0, 0] + bot_hor[1, 0])/2.0
    bot_center_d = (bot_ver[0, 0] + bot_ver[1, 0])/2.0
    
    print(top_center_w)
    print(bot_center_w)
    print(top_center_h)
    print(bot_center_h)
    
    angle_hor = np.arctan((top_center_w - bot_center_w) / (top_center_h - bot_center_h))
    angle_ver = np.arctan((top_center_d - bot_center_d)/(top_center_h - bot_center_h))
    print(angle_hor)    #probs in radian
    print(angle_ver)
    
    
    from scipy.ndimage.interpolation import rotate
    # map3d_new = rotate(map3d, angle = 90.0)
    
    
    
    def foo(a):
        angle = -angle_hor*180/np.pi
        return rotate(a, angle=angle, prefilter = False)
    def foo2(a):
        angle = -angle_ver*180/np.pi
        # As prefilter do SINC filter in Fourier domain (quench high frequency components)
        return rotate(a, angle=angle, prefilter = False)
    # map3d_new = rotate(map3d, angle=90.

    c = np.arange(0, 1937) # (0, 1937)
    lambda0 = lambda : np.stack([foo(map3d[:, :, c_i]) for c_i in c], axis = 2)

    map3d_new = time_func(lambda0)

    c = np.arange(0, 1951)    # (0, 1951)
    lambda0 = lambda : np.stack([foo2(map3d_new[:, c_i, :]) for c_i in c], axis = 1)
    map3d_new = time_func(lambda0)
    
    return map3d_new



def norm_imshow(im):
    plt.imshow(im, vmin = 0, vmax = bit_max)

# TODO, removes top and bottom pill
def cropper():
    ...
  
    
def cross_section(map3d, norm_im = True):
    shape = np.shape(map3d)
    hor_index = (shape[2]//2)
    ver_index = (shape[1] // 2)
    
    cross_hor = map3d[:, :, hor_index]
    cross_ver = map3d[:, ver_index, :]
    
    plt.subplot(2, 1, 1)
    if norm_im:
        norm_imshow(cross_hor)
    else:
        plt.imshow(cross_hor, vmin = 0)
    plt.title('horizontal cross section')
    plt.subplot(2, 1, 2)
    if norm_im:
        norm_imshow(cross_ver)
    else:
        plt.imshow(cross_ver, vmin = 0)
    plt.title('vertical cross section')
    plt.show()
    
if __name__ == '__main__':
    main()