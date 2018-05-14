import matplotlib.pyplot as plt
import numpy as np

from link_to_soliton.paint_tools import image_tools


def main():
    #settings:
    version = 0
    name_ext_no_format = '_v{0}_h{1}_w{2}_out_f{3}_gap{4}.bmp'
    
    name_in = '19_hand_clean.tif'
    name_base = 'hand_inpainting'
    
    if 0:
        folder_in = '/home/lameeus/data/ghent_altar/roman/input/'
        folder_crops = '/home/lameeus/data/ghent_altar/roman/inpainting/'
        # folder_out = '/home/lameeus/data/ghent_altar/roman/stitch/'
    elif 0:
        folder_in = '/home/lameeus/data/ghent_altar/input/19_hand/'
        folder_crops = '/home/lameeus/data/ghent_altar/inpainting/hand/'
        folder_out = '/home/lameeus/data/ghent_altar/inpainting/stitch/'
    elif 0:
        folder_in = '/home/lameeus/data/ghent_altar/input/hierarchy/19_small/'
        folder_crops = '/home/lameeus/data/ghent_altar/inpainting/hand/unet/f2_gap3/'
        folder_out = '/home/lameeus/data/ghent_altar/output/hierarchy/19_small/te_unet/'

        name_in = 'clean.tif'
        name_base = 'output_tijana'
        version = -1
        name_ext_no_format = '_h{1}_w{2}_out_f{3}_gap{4}.bmp'
    elif 1:
        folder_in = '/home/lameeus/data/ghent_altar/input/hierarchy/19_big/'
        folder_crops = '/home/lameeus/data/ghent_altar/inpainting/evangelist/unet/f2_gap3/'
        folder_out = '/home/lameeus/data/ghent_altar/output/hierarchy/19_big/te_unet/'

        name_in = 'clean.tif'
        name_base = 'output_tijana'
        version = -1
        name_ext_no_format = '_h{1}_w{2}_out_f{3}_gap{4}.bmp'
    
        
    img_in = image_tools.path2im(folder_in + name_in)
    
    if 1:
        plt.figure()
        plt.imshow(img_in)
        plt.show(False)

    shape = np.shape(img_in)
    
    img_new = np.ones(shape) * 0.5
    
    # Specs should be correct!
    if version == 0:
        if 0:
            folder_crops += 'v0/'
            f = 6
            gap = 4
            w_crop = 400
            ext = 2*f
        else:
            folder_crops += 'v0/'
            f = 4
            gap = 2
            w_crop = 200
            ext = 2 * f
    
    elif version == 4:
        folder_crops += 'v4/'
        f = 4
        gap = 2
        w_crop = 100
        ext = 5
        
    elif version == -1:
        f = 2
        gap = 3
        w_crop = 100
        ext = 2*f
        
    n_h = np.ceil(shape[0]/w_crop).astype(int)
    n_w = np.ceil(shape[1]/w_crop).astype(int)
    
    for i_h in range(n_h):
        for i_w in range(n_w):
            name_ext = name_ext_no_format.format(version, i_h+1, i_w+1, f, gap)
            name_full = folder_crops + name_base + name_ext
            
            try:
                rgb_crop = image_tools.path2im(name_full)
            except:
                print('No file: {}'.format(name_full))
                continue
            
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
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(img_in)
        plt.subplot(1,2,2)
        plt.imshow(img_new)
        plt.show()
        
        if 1:
            image_tools.save_im(img_new, folder_out + 'inpainting_full_f{}_g{}.png'.format(f, gap))

if __name__ == '__main__':
    main()
