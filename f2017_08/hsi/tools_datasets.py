""" for predefined datasets
"""

import numpy as np

from f2017_08.hsi import tools_data


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

def hsi_mask():
    if 1:
        from link_to_soliton.paint_tools import image_tools
        mask_img = image_tools.path2im('/home/lameeus/data/hsi/mask.png', type = 'png')
    
        shape = np.shape(mask_img)
        mask = np.zeros(shape = (shape[0], shape[1]), dtype=int)
    
        mask[mask_img[:,:,0] < 0.5] = 1
        
        return mask
