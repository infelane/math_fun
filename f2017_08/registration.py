from link_to_soliton.paint_tools import image_tools
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D


def example():
    folder = '/home/lameeus/data/ghent_altar/input/'
    im1 = image_tools.path2im(folder + '19_clean.tif')
    im2 = image_tools.path2im(folder + '19_rgb.tif')[:, :, 0:3]
    # im2 = image_tools.path2im(folder + '19_ir.tif')[:, :, 0:3]
    
    reg1(im1, im2)


def reg1(im1, im2):
    # # for IR
    # expand_h = 2.742140625 * 0.9965318627450981
    # expand_w = 2.72565557753 * 1.0002197802197803#2.71565837363
    # shift_h = -11 #-11
    # shift_w = -17 #-11
    
    # for RGB
    expand_h = 2.74207569428
    expand_w = 2.72543070106 #2.71565837363
    shift_h = -11 #-11
    shift_w = -11 #-11
    
    if 0:
        plt.figure()
        imshow(im1)
    
        plt.figure()
        imshow(im2)

    # First y (height) then x (width)
    # im2_reshape = apply_change(im2, 2.745, 2.727, rot = -0.1) # 2.73747126457
    im2_reshape = apply_change(im2, expand_h, expand_w, rot = 0) # 2.73747126457
    # im_overlay = overlay1(im1, im2_reshape, shift_h, shift_w)
    im_overlay, im2_shifted = overlay2(im1, im2_reshape, shift_h, shift_w)
    
    if 0:
        folder = '/home/lameeus/data/ghent_altar/input/registration/'
        image_tools.save_im(im2_shifted, folder + '19_rgb_reg.tif', check_prev=False)
    
    if 0:
        plt.figure()
        plt.subplot(1, 3, 1)
        imshow(im1)
        plt.subplot(1, 3, 2)
        # imshow(im2_reshape)
        imshow(im2_shifted)
        plt.subplot(1, 3, 3)
        imshow(im_overlay)
        plt.show()
    
    h_center = 850
    h_delta = 50
    w_center = 800
    w_delta = 50
    
    im3_zoom = im_overlay[h_center - h_delta: h_center + h_delta, w_center - w_delta: w_center + w_delta, :]
    # im3_zoom = im_overlay[...]
    
    highpass1 = high_pass(im1)
    highpass2 = high_pass(im2_reshape)

    highpass_overlay = overlay1(highpass1, highpass2, shift_h, shift_w)
    # high_overlay_zoom = highpass_overlay[h_center - h_delta: h_center + h_delta, w_center - w_delta: w_center + w_delta, :]
    high_overlay_zoom = highpass_overlay[...]
    
    if 1:
        calc_conv(highpass1, highpass2, shift_h, shift_w)
    
    if 1:
        plt.figure()
        plt.subplot(2, 2, 1)
        imshow(im3_zoom)
        
        plt.subplot(2, 2, 2)
        imshow(high_overlay_zoom)

        plt.subplot(2, 2, 3)
        imshow(highpass1)
        plt.title('high pass 1')
        
        plt.subplot(2, 2, 4)
        imshow(highpass2)
        plt.title('high pass 2')
        
        plt.show()
    
    return


def calc_conv(im1, im2, shift_h, shift_w):
    shape1 = np.shape(im1)
    shape2 = np.shape(im2)
    
    para = []
    
    # # x = []
    # # y = []
    # # z = []
    # #
    # # from mpl_toolkits.mplot3d import axes3d
    # # import matplotlib.pyplot as plt
    # #
    # # fig = plt.figure()
    # # ax = fig.add_subplot(111, projection='3d')
    # #
    # # # Grab some test data.
    # # X, Y, Z = axes3d.get_test_data(0.05)
    # #
    # # # Plot a basic wireframe.
    # # ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
    # #
    # # plt.show()
    # #
    # # print(X)
    # # print(Y)
    # # print(Z)
    # # print(np.shape(X))
    # # print(np.shape(Z))
    #
    # n = 6
    # x = np.zeros((n, n))
    # y = np.zeros((n, n))
    # z = np.zeros((n, n))
    #
    # delta_range = np.arange(-n//2, n - n//2)
    #
    # for i in range(0, n):
    #     for j in range(0, n):
    #
    #         delta_h = 19 + delta_range[i]
    #         delta_w = 0 + delta_range[j]
    #
    #         # top = max(0, delta_h)
    #         # bot = max(shape1[0], shape2[0] + delta_h)
    #         # left = max(0, delta_w)
    #         # right = max(shape1[1], shape2[1] + delta_w)
    #         #
    #         # tot_h = bot - top
    #         # tot_w = right - left
    #
    #         im1_crop = im1[max(0, delta_h): min(shape1[0], shape2[0] + delta_h),
    #                    max(0, delta_w): min(shape1[1], shape2[1] + delta_w)]
    #
    #         im2_crop = im2[max(0, -delta_h): min(shape2[0], shape1[0] - delta_h),
    #                    max(0, -delta_w): min(shape2[1], shape1[1] - delta_w)]
    #
    #         conv = im1_crop*im2_crop
    #
    #         sum_conv = np.sum(conv)
    #         para.append([delta_h, delta_w, sum_conv])
    #
    #         x[i, j] = delta_h
    #         y[i, j] = delta_w
    #         z[i, j] = sum_conv
    #
    #         # x.append(delta_h)
    #         # y.append(delta_w)
    #         # z.append(sum_conv)
    #
    #         # plt.figure()
    #         # plt.imshow(conv, vmin = 0., vmax = 1.)
    #         # plt.title('{}'.format(sum_conv))
    #
    #         #
    #         # im_mix = np.zeros(shape=(tot_h, tot_w, 3))
    #         #
    #         # print(np.shape(im_mix))
    #         #
    #         # # im_mix[-top : -top + shape1[0], -left : -left + shape1[1], 0] =\
    #         # #     np.mean(self.im1, axis = 2)
    #         # # im_mix[-top + delta_h: -top + delta_h + shape2[0], -left + delta_w: -left + delta_w + shape2[1], 2] =\
    #         # #     np.mean(im2_reshape, axis=2)
    #         #
    #         # im_mix[-top: -top + shape1[0], -left: -left + shape1[1], 0:1] = \
    #         #     im1[:, :, 0:1]
    #         # im_mix[-top + delta_h: -top + delta_h + shape2[0], -left + delta_w: -left + delta_w + shape2[1], 1:3] = \
    #         #     im2[:, :, 1:3]
    #
    # # plt.show()
    # #
    # # x, y, z = zip(*para)
    #
    # # xy = zip(x, y)
    #
    # # z = np.reshape(np.array(z), (6, 6))
    #
    # # plt.plot(x, z)
    # # plt.show()
    
    spacing = 400
    h_start = spacing
    h_end = shape1[0] - spacing
    w_start = spacing
    w_end = shape1[1] - spacing
    
    square_h = np.arange(h_start, h_end, spacing)
    square_w = np.arange(w_start, w_end, spacing)
    len_h = len(square_h)
    len_w = len(square_w)

    n = 21
    
    info_h = np.zeros((len_h, len_w))
    info_w = np.zeros((len_h, len_w))
    para_all = np.zeros((len_h, len_w, n, n))

    info_delta_h = np.zeros((len_h, len_w))
    info_delta_w = np.zeros((len_h, len_w))
    
    delta_h_mean = shift_h
    delta_w_mean = shift_w

    n_half = n // 2
    n_delta = 2
    
    delta_range = np.arange(-n_half*n_delta, n*n_delta - n_half*n_delta, n_delta)
    for i in range(len_h):
        for j in range(len_w):
            h_ij = square_h[i]
            w_ij = square_w[j]
    
            x = np.zeros((n, n))
            y = np.zeros((n, n))
            z = np.zeros((n, n))

            im1_square = im1[h_ij:h_ij + spacing, w_ij:w_ij + spacing]

            for i_delta_h in range(0, n):
                for i_delta_w in range(0, n):
                    delta_h = delta_h_mean + delta_range[i_delta_h]
                    delta_w = delta_w_mean + delta_range[i_delta_w]
            
                    im2_square = im2[h_ij+ delta_h:h_ij + spacing + delta_h, w_ij+ delta_w:w_ij + spacing + delta_w]

                    conv = im1_square * im2_square
                    sum_conv = np.sum(conv)

                    para_all[i, j, i_delta_h, i_delta_w] = sum_conv

                    x[i_delta_h, i_delta_w] = delta_h
                    y[i_delta_h, i_delta_w] = delta_w
                    z[i_delta_h, i_delta_w] = sum_conv

            z_argmax = np.unravel_index(z.argmax(), z.shape)
            
            # if x[z_argmax] == delta_h_mean + delta_range[0] or x[z_argmax] == delta_h_mean + delta_range[-1]:
            #     plt.imshow(z)
            #     plt.show()
                
            info_delta_h[i, j] = x[z_argmax]
            info_delta_w[i, j] = y[z_argmax]
            
            info_h[i, j] = i
            info_w[i, j] = j

    z = para_all[:, :, n_half, n_half]
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # Plot a basic wireframe.
    # ax.plot_wireframe(info_h, info_w, z, rstride=1, cstride=1)
    # plt.xlabel('square i')
    # plt.ylabel('square j')
    # # plt.show()

    plt.figure()
    imshow(z)
    plt.colorbar()
    plt.xlabel('square i')
    plt.ylabel('square j')
    plt.title('amount of information')
    # plt.show()
    
    delta_h_min = delta_h_mean + delta_range[0]
    delta_h_max = delta_h_mean + delta_range[-1]
    delta_w_min = delta_w_mean + delta_range[0]
    delta_w_max = delta_w_mean + delta_range[-1]

    plt.figure()
    plt.imshow(info_delta_h, vmin= delta_h_min, vmax = delta_h_max)
    plt.colorbar()
    plt.xlabel('square i')
    plt.ylabel('square j')
    plt.title('delta h best')
    # plt.show()

    w_medians = np.median(info_delta_w, axis = 0)
    square_w_center = square_w + spacing/2
    m, b = np.polyfit(square_w_center, w_medians, 1)
    print('{} {}'.format(m, b))
    print('w expand * {}\nshift w = {}'.format(1-m, b))
    
    h_medians = np.median(info_delta_h, axis = 1)
    square_h_center = square_h + spacing/2
    m, b = np.polyfit(square_h_center, h_medians, 1)
    print('{} {}'.format(m, b))
    print('h expand * {}\nshift h = {}'.format(1-m, b))
    
    plt.figure()
    plt.imshow(info_delta_w, vmin= delta_w_min, vmax = delta_w_max)
    plt.colorbar()
    plt.xlabel('square i')
    plt.ylabel('square j')
    plt.title('delta w best')

    delta_h_median = np.median(info_delta_h)
    delta_w_median = np.median(info_delta_w)
    
    print('h: {}, w: {}'.format(delta_h_median, delta_w_median))

    arrow_analysis(info_delta_h, info_delta_w)
    
    plt.figure()
    plt.quiver(np.transpose(info_delta_h) - delta_h_median, np.transpose(info_delta_w) - delta_w_median)
    plt.xlabel('h')
    plt.ylabel('w')
    plt.show()
    
    # z_argmax = np.unravel_index(z.argmax(), z.shape)
    # print('h:{} w:{}'.format(x[z_argmax], y[z_argmax]))
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # Plot a basic wireframe.
    # ax.plot_wireframe(x, y, z, rstride=1, cstride=1)
    # plt.xlabel('delta h')
    # plt.ylabel('delta w')
    #
    # plt.show()
    
    

def arrow_analysis(info_delta_h, info_delta_w):
    
    shape = np.shape(info_delta_h)
    
    rot_all = np.zeros(shape)
    rad_all = np.zeros(shape)
    
    
    
    for i in range(1, shape[0]-1):
        for j in range(1, shape[1] - 1):
            
            center_dir = [info_delta_h[i, j], info_delta_w[i, j]]
            
            rot_ij = 0
            rad_ij = 0
            
            # left, up, right, bot
            
            rot_ij += -info_delta_h[i, j+1] - center_dir[0]
            rad_ij += info_delta_w[i, j+1] - center_dir[1]

            rot_ij += -info_delta_w[i-1, j] - center_dir[1]
            rad_ij += -info_delta_h[i-1, j] - center_dir[0]

            rot_ij += info_delta_h[i, j - 1] - center_dir[0]
            rad_ij += -info_delta_w[i, j - 1] - center_dir[1]

            rot_ij += info_delta_w[i+1, j] - center_dir[1]
            rad_ij += info_delta_h[i+1, j] - center_dir[0]

            rot_all[i, j] = rot_ij/4
            rad_all[i, j] = rad_ij/4
            
    plt.figure()
    plt.subplot(1,2,1)
    imshow(rot_all)
    plt.colorbar()
    plt.title('rotation : pixels per spacing')
    plt.subplot(1,2,2)
    imshow(rad_all)
    plt.colorbar()
    plt.title('reshaping : pixels per spacing')
    plt.show()
    

def imshow(im):
    plt.imshow(im, interpolation = 'nearest')


def high_pass(im):
    # blurred = gaussian_filter(im, sigma=7)
    
    # stack_blurred = [gaussian_filter(im[:,:,i], sigma=7) for i in range(3) ]
    
    sigma = 15

    blurred = cv2.GaussianBlur(im,(sigma,sigma),0)
    # blurred = cv2.blur(im, (sigma, sigma), 0)
    
    abs_diff = np.mean(np.abs(im - blurred), axis = 2)
    max_val = np.max(abs_diff)
    print(max_val)
    norm = abs_diff/max_val
    
    # return 0.5 + im - blurred)
    # return np.stack([norm, norm, norm], axis = 2)
    return norm
    

def apply_change(im2, rescale_h, rescale_w = None, rot = 0):
    if rescale_w is None:
        rescale_w = rescale_h
    
    if rot == 0:
        im2_rot = im2[...]

    if rot != 0:
        rows, cols, ch = np.shape(im2)
    
        M = cv2.getRotationMatrix2D((rows / 2, cols / 2), rot, 1)
        im2_rot = cv2.warpAffine(im2, M, (cols, rows))
        
    im2_reshape =  cv2.resize(im2_rot, (0, 0), fx=rescale_w, fy=rescale_h, interpolation=4)
    
    return im2_reshape


def overlay1(im1, im2, delta_h_in, delta_w_in):
    delta_h = -delta_h_in
    delta_w = -delta_w_in
    
    shape1 = np.shape(im1)
    shape2 = np.shape(im2)
    
    top = min(0, delta_h)
    bot = max(shape1[0], shape2[0] + delta_h)
    left = min(0, delta_w)
    right = max(shape1[1], shape2[1] + delta_w)
    
    tot_h = bot - top
    tot_w = right - left
    
    im_mix = np.zeros(shape=(tot_h, tot_w, 3))
    
    print(np.shape(im_mix))
    
    # im_mix[-top : -top + shape1[0], -left : -left + shape1[1], 0] =\
    #     np.mean(self.im1, axis = 2)
    # im_mix[-top + delta_h: -top + delta_h + shape2[0], -left + delta_w: -left + delta_w + shape2[1], 2] =\
    #     np.mean(im2_reshape, axis=2)
    
    if len(shape1) == 3:
        im1_partial = im1[:, :, 0]
    elif len(shape1) == 2:
        im1_partial = im1[...]
      
    if len(shape2) == 3:
        im2_partial1 = im2[:, :, 1]
        im2_partial2 = im2[:, :, 2]
    elif len(shape2) == 2:
        im2_partial1 = im2[...]
        im2_partial2 = im2[...]
        
    
    im_mix[-top: -top + shape1[0], -left: -left + shape1[1], 0] = \
        im1_partial
    im_mix[-top + delta_h: -top + delta_h + shape2[0], -left + delta_w: -left + delta_w + shape2[1], 1] = \
        im2_partial1
    im_mix[-top + delta_h: -top + delta_h + shape2[0], -left + delta_w: -left + delta_w + shape2[1], 2] = \
        im2_partial2
    
    # im_mix[:,:,1] = 1

    return im_mix



def overlay2(im1, im2, delta_h_in, delta_w_in):
    """im 2 recut to be registrated on im1
    """
    
    delta_h = -delta_h_in
    delta_w = -delta_w_in
    
    shape1 = np.shape(im1)
    shape2 = np.shape(im2)
    
    if delta_h > 0:
        h0_new = delta_h
        h0_old = 0
    else:
        h0_new = 0
        h0_old = -delta_h
        
    if delta_w > 0:
        w0_new = delta_w
        w0_old = 0
    else:
        w0_new = 0
        w0_old = -delta_w
        
    if delta_h + shape2[0] > shape1[0]:
        h1_new = shape1[0]
        h1_old = shape1[0] - delta_h
    else:
        h1_new = shape2[0] + delta_h
        h1_old = shape2[0]
        
    if delta_w + shape2[1] > shape1[1]:
        w1_new = shape1[1]
        w1_old = shape1[1] - delta_w
    else:
        w1_new = shape2[1] + delta_w
        w1_old = shape2[1]
        
    # bot = max(shape1[0], shape2[0] + delta_h)
    # left = min(0, delta_w)
    # right = max(shape1[1], shape2[1] + delta_w)
    #
    # tot_h = bot - top
    # tot_w = right - left
    #
    
    im2_new = np.ones(shape=(shape1[0], shape1[1], 3)) * 0.5
    im_mix = np.zeros(shape=(shape1[0], shape1[1], shape2[2]))

    # im2_new[a:b, c:d, :] = im2[e:f, g:h, :]
    
    delta = 3000
    
    im2_new[h0_new:h1_new, w0_new:w1_new, :] = im2[h0_old: h1_old, w0_old:w1_old, :]
    
    # print(np.shape(im_mix))
    #
    # # im_mix[-top : -top + shape1[0], -left : -left + shape1[1], 0] =\
    # #     np.mean(self.im1, axis = 2)
    # # im_mix[-top + delta_h: -top + delta_h + shape2[0], -left + delta_w: -left + delta_w + shape2[1], 2] =\
    # #     np.mean(im2_reshape, axis=2)
    #
    # if len(shape1) == 3:
    #     im1_partial = im1[:, :, 0]
    # elif len(shape1) == 2:
    #     im1_partial = im1[...]
    #
    # if len(shape2) == 3:
    #     im2_partial1 = im2[:, :, 1]
    #     im2_partial2 = im2[:, :, 2]
    # elif len(shape2) == 2:
    #     im2_partial1 = im2[...]
    #     im2_partial2 = im2[...]
    
    im_mix[:, :, 0:1] = im1[:,:,0:1]
    im_mix[:, :, 1] = im2_new[:, :, -1]
    im_mix[:, :, 2] = im2_new[:, :, -1]

    return im_mix, im2_new


if __name__ == '__main__':
    example()
    