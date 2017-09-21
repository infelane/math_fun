import matplotlib.pyplot as plt
import numpy as np

from link_to_soliton.paint_tools import image_tools


def main():
    folder_in = '/home/lameeus/data/ghent_altar/roman/input/'
    folder = '/home/lameeus/data/ghent_altar/roman/inpainting/'
    folder_out = '/home/lameeus/data/ghent_altar/roman/stitch/'
    
    name_in = '16_VIS_reg.tif'
    
    name_base = 'roman_inpainting'

    img_in = image_tools.path2im(folder_in + name_in)
    
    #settings:
    version = 4
    
    if 0:
        plt.imshow(img_in)
        plt.show()

    shape = np.shape(img_in)
    
    img_new = np.ones(shape) * 0.5
    
    if version == 4:
        f = 4
        gap = 2
        w_crop = 100
        ext = 5
        
    n_h = np.ceil(shape[0]/w_crop).astype(int)
    n_w = np.ceil(shape[1] / w_crop).astype(int)
    
    for i_h in range(n_h):
        for i_w in range(n_w):
            
            name_ext = '_v{}_h{}_w{}_out_f{}_gap{}.bmp'.format(version, i_h+1, i_w+1, f, gap)
            name_full = folder + name_base + name_ext
            rgb_crop = image_tools.path2im(name_full)
            
            if 0:
                print(name_full)
                
            if 0:
                shape_crop = np.shape(rgb_crop)
                print(shape_crop)
           
            # Cut away border, exceptions for when i = n_h or 0
            if i_h == 0:
                h0 = 0
            else:
                rgb_crop = rgb_crop[ext + 1:, :, :]
                h0 = (i_h) * w_crop

            if i_w == 0:
                w0 = 0
            else:
                rgb_crop = rgb_crop[:, ext + 1:, :]
                w0 = (i_w) * w_crop
     
            if i_h == n_h-1:
                shape_crop = np.shape(rgb_crop)
                h1 = h0 + shape_crop[0]
            else:
                rgb_crop = rgb_crop[:w_crop, :, :]
                h1 = h0 + w_crop

            if i_w == n_w-1:
                shape_crop = np.shape(rgb_crop)
                w1 = w0 + shape_crop[1]
            else:
                rgb_crop = rgb_crop[:, :w_crop, :]
                w1 = w0 + w_crop
                
            if 0:
                shape_crop = np.shape(rgb_crop)
                print(shape_crop)
              
            img_new[h0:h1, w0:w1, :] = rgb_crop
    
    if 1:
        plt.subplot(1,2,1)
        plt.imshow(img_in)
        plt.subplot(1,2,2)
        plt.imshow(img_new)
        plt.show()
        
        if 1:
            image_tools.save_im(img_new, folder_out + 'inpainting_full_v{}.png'.format(version))

if __name__ == '__main__':
    main()
