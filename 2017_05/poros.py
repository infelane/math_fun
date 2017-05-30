# guess porosity of tablets

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

bit_max = 65536 - 1     # 16bit


def open_orig():
    def gen_3d():
        n_begin = 500  # 500 to 1700
        n_end = 1700
        
        folder = '/net/ipids/microscopy/CT/pharmacy_tablets/'
        
        i_im = np.arange(n_begin, n_end + 1, dtype=int)
        
        def foo(index):
            im_name_i = folder + '3_{0:05d}.tif'.format(index)
            im = Image.open(im_name_i)
            imarray = np.array(im, dtype=np.uint16)
            return imarray
        
        data_all = np.stack([foo(index) for index in i_im], axis=0)
        
        return data_all
    
    # name_3d = '3dimage'
    
    data_all = time_func(gen_3d)  # 14s
    
    return data_all


def open_rot():
    folder = '/scratch/lameeus/data/pharmacy_tablets/rotated/'

    def gen_3d():
        def foo(index):
            im_name_i = folder + 'cut_{}.tif'.format(index)
            im = Image.open(im_name_i)
            imarray = np.array(im, dtype=np.uint16)
            return imarray
    
        # nr_images = 1291 # 1291 max
        nr_images = 1291
    
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


def main():
    if 0: # if you want to rerotate the images
        data_all = open_orig()
        data_rot = rotation(data_all)

        if 0:
            plot_inter(data_all)

        if 1:
            cross_section(data_rot)
            plot_inter(data_rot)
    
        save_interstate(data_rot)

    else: # load rotated image
        data_rot = open_rot()

        # cross_section(data_rot)

        filter_cross_section(data_rot)
   
    
    # cross_section(data_all)
    #
    # data_rot = rotation(data_all)
    # cross_section(data_rot)
    #
    #
    # intens, bins = np.histogram(data_all, bins = 256, range=[0, bit_max])
    #
    #
    #
    # # print(a)
    #
    # bins_center = (bins[0:-1] + bins[1:])/2.0
    #
    # print(np.shape(bins_center))
    # print(np.shape(intens))
    #
    # plt.plot(bins_center, intens)
    # plt.show()
    

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
    
    
folder_name = '/scratch/lameeus/data/pharmacy_tablets/'
def save_interstate(map3d, folder_name= folder_name):
    
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
    
    # # cross_section(map3d)
    #
    # shape = np.shape(map3d)
    #
    # h_bot = 1120
    # h_top = 232
    #
    #
    # h_now = 100
    #
    # start = time.time()
    #


    #
    # if dist_inner <= 1.0 or 1: # TODO
    #     # kernel overlaps with center, do something else
    #

    #
    #
    #
    #
    #     # TODO
    #     # center_filter = [center[0] - w_pixel[0], center[1] - d_pixel[0]]
    #     # print(center_filter)
    #
    #
    #     shape_filter = np.shape(filter_all)
    #

    #

    #

    #

    #
    #
    #
    # else: # TODO speeds up the filter
    #     border_inner_w = center[0] - dist_inner
    #
        
        
    # TODO start from here, clean up top
    
    # Settings
    # step_size = 10
    # r_range = np.arange(0., 850., step_size)
    # h_range = np.arange(100, 1100, step_size, dtype=int)
    # std = step_size     # 10.0
    step_size = 5
    r_range = np.arange(0., 850., step_size, dtype=float)
    h_range = np.arange(100, 1100, step_size, dtype=int)
    std = 5.0     # 10.0

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
    
    results = results.astype(np.uint16)
    im = Image.frombytes('I;16', (r_len, h_len), results.tobytes())
    im.save("results.tif")

    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=0, vmax=bit_max)
    im = cmap(norm(results))
    plt.imsave('colormap.png', im)
    
    plt.imshow(results, vmin = 0.0, vmax = bit_max)
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