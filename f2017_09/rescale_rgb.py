# Just a simple script to rescale the RGB image to the be close enough registered to HSI data

import cv2
import matplotlib.pyplot as plt
import numpy as np

from link_to_soliton.paint_tools import image_tools
from f2017_08 import hsi_main


def main():
    folder = '/home/lameeus/data/hsi/max/'
    file = 'ABouts_PortretMan_VIS.tif'
    rgb_orig = image_tools.path2im(folder + file)
    
    folder = '/ipi/research/lameeus/data/hsi/'
    file = 'hsi_rgb.png'
    hsi_rgb = image_tools.path2im(folder + file)
    
    if 1:
        image_tools.save_im((hsi_rgb-0.5)*0.5+0.5, '/ipi/research/lameeus/data/hsi/hsi_rgb_grey.png')
    
    def show_prog(a, bool_mix = False):
        plt.figure()
        plt.subplot(1,3, 1)
        plt.imshow(hsi_rgb)
        plt.subplot(1, 3, 2)
        plt.imshow(a)
        
        if bool_mix:
            mix = np.copy(hsi_rgb)
            mix[:,:,0] = a[:,:,0]
            plt.subplot(1, 3, 3)
            plt.imshow(mix)
        
        plt.show()
        
    # initial crop
    crop = rgb_orig[400:5286, 3648:7252, :]
    
    hsi_p1_h = 218
    hsi_p1_w = 320

    hsi_p2_h = 2490
    hsi_p2_w = 1707

    rgb_p1_h = 534
    rgb_p1_w = 574
    
    rgb_p2_h = 4381
    rgb_p2_w = 2985
    
    delta_hsi_h = hsi_p2_h - hsi_p1_h
    delta_hsi_w = hsi_p2_w - hsi_p1_w
    delta_hsi = np.sqrt(np.square(delta_hsi_h) + np.square(delta_hsi_w))
    
    delta_rgb_h = rgb_p2_h - rgb_p1_h
    delta_rgb_w = rgb_p2_w - rgb_p1_w
    delta_rgb = np.sqrt(np.square(delta_rgb_h) + np.square(delta_rgb_w))
    
    print(delta_hsi_h/delta_rgb_h)
    print(delta_hsi_w / delta_rgb_w)
   

    rescale = delta_hsi / delta_rgb
    print(rescale)

    im2_reshape =  cv2.resize(crop, (0, 0), fx=rescale, fy=rescale, interpolation=4)

    # new_rgb = cv2.flip(rgb_orig, 1)
    
    if 0:
        show_prog(crop)
        
    if 0:
        show_prog(im2_reshape)
    
    shape = np.shape(hsi_rgb)
    
    h0 = 83-7
    w0 = 36-7
    
    h1 = h0 + shape[0]
    w1 = w0 + shape[1]

    main_data = hsi_main.main_data()
    mask = main_data['mask_annot']

    plt.figure()
    plt.imshow(mask)
    
    crop2 = im2_reshape[h0:h1, w0:w1,:]
    
    crop2_copy = np.copy(crop2)
    crop2_copy[mask == 1, :] = 1.
    show_prog(crop2_copy, True)
    
    if 0:
        image_tools.save_im(crop2, '/home/lameeus/data/hsi/rgb_registrated.png')


if __name__ == '__main__':
    main()
    