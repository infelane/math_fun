""" for predefined datasets
"""

import numpy as np

from f2017_08.hsi import tools_data
from link_to_soliton.paint_tools import image_tools


def hsi_raw():
    """
    hyperspectral image dataset from small painting, for Max!
    :return:
    """
    
    import spectral.io.envi as envi
    
    folder = '/ipi/research/ljilic/study/study__artwork_hsi_2017/images/portret_man_20170727_imec/'
    # TESTED! other images are not important

    name_header = 'painting_corrected.hdr'
    name_image = 'painting_corrected.raw'

    lib = envi.open(folder + name_header, image=folder + name_image)

    img = np.array(lib[:, :, :])
    wavelengths = np.array(lib.bands.centers)
    max_value = lib.params().metadata['max value']
    
    if 0:
        print(np.max(wavelengths))
        print(np.min(wavelengths))
    
    return {'img' : img, 'wavelengths' : wavelengths, 'max' : max_value}

def hsi_processed():
    folder = '/home/lameeus/data/hsi/'
    file = folder + 'img_norm.npy'
    
    if 0:
        # takes 26s
        
        hsi_data = hsi_raw()
        img = hsi_data['img']
        img_norm = tools_data.norm_hsi(img)

        np.save(file, img_norm)
        
    else:
        # Takes
        img_norm = np.load(file)
    
    return img_norm


def hsi_annot():
    folder_save = '/home/lameeus/data/hsi/'
    
    if 0:
        folder_annot = '/ipi/research/lameeus/data/hsi/'
        
        file = folder_annot + 'annot.png'
        img = image_tools.path2im(file)[..., 0:3]
       
        shape = np.shape(img)
        shape_annot = [shape[0], shape[1], 6]
        img_annot = np.zeros(shape = shape_annot)
        
        r0 = np.equal(img[:,:,0], 0)
        r1 = np.equal(img[:, :, 0], 1)
        g0 = np.equal(img[:,:,1], 0)
        g1 = np.equal(img[:, :, 1], 1)
        b0 = np.equal(img[:,:,2], 0)
        b1 = np.equal(img[:, :, 2], 1)

        red = np.logical_and(np.logical_and(r1, g0), b0)
        green = np.logical_and(np.logical_and(r0, g1), b0)
        blue = np.logical_and(np.logical_and(r0, g0), b1)
        cyan = np.logical_and(np.logical_and(r0, g1), b1)
        yellow = np.logical_and(np.logical_and(r1, g1), b0)
        magenta = np.logical_and(np.logical_and(r1, g0), b1)

        # white = np.logical_and(np.logical_and(r1, g0), b0)

        # red = first*(1-second)*(1-third)
        # green = (1-first) * (second) * (1 - third)
        # red_map = )
        # print(np.shape(red_map))
        # img_annot[red, 0] = 1
        # img_annot[green, 1] = 1
        # green_map = np.equal(img, [0, 1, 0])
        # img_annot[red_map, 1] = 1

        img_annot[red, 0] = 1
        img_annot[green, 1] = 1
        img_annot[blue, 2] = 1
        img_annot[cyan, 3] = 1
        img_annot[yellow, 4] = 1
        img_annot[magenta, 5] = 1
        
        np.save(folder_save + 'y_annot.npy', img_annot)
        
    else:
        img_annot = np.load(folder_save + 'y_annot.npy')
        
    return img_annot
    

def hsi_mask():
    if 1:
        from link_to_soliton.paint_tools import image_tools
        mask_img = image_tools.path2im('/home/lameeus/data/hsi/mask.png', type = 'png')
    
        shape = np.shape(mask_img)
        mask = np.zeros(shape = (shape[0], shape[1]), dtype=int)
    
        mask[mask_img[:,:,0] < 0.5] = 1
        
        return mask
