""" for predefined datasets
"""

import numpy as np

from f2017_08.hsi import tools_data
from link_to_soliton.paint_tools import image_tools


def hsi_raw(folder=None, name_header=None, name_image=None):
    """
    hyperspectral image dataset from small painting, for Max!
    :return:
    """
    
    if folder is None:
        folder = '/ipi/research/ljilic/study/study__artwork_hsi_2017/images/portret_man_20170727_imec/'

    if name_header is None:
        name_header = 'painting_corrected.hdr'
    if name_image is None:
        name_image = 'painting_corrected.raw'

    import spectral.io.envi as envi
    
    # TESTED! other images are not important

    lib = envi.open(folder + name_header, image=folder + name_image)

    img = np.array(lib[:, :, :])
    img = np.flip(img, axis = 1)    # had to be flipped horizontally, like the RGB image
    
    wavelengths = np.array(lib.bands.centers)
    max_value = lib.params().metadata['max value']
    
    if 0:
        print(np.max(wavelengths))
        print(np.min(wavelengths))
    
    return {'img' : img, 'wavelengths' : wavelengths, 'max' : max_value}

def hsi_processed():
    folder = '/home/lameeus/data/hsi/'
    file = folder + 'img_norm.npy'
    
    if 0:   # Changes back to 0
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
    
    if 0:   # TODO SET AT 0 after done
        folder_annot = '/ipi/research/lameeus/data/hsi/'
        
        file = folder_annot + 'annot2.png'
        img = image_tools.path2im(file)[..., 0:3]
       
        shape = np.shape(img)
        shape_annot = [shape[0], shape[1], 8]
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
        white = np.logical_and(np.logical_and(r1, g1), b1)
        black = np.logical_and(np.logical_and(r0, g0), b0)
        
        if 0:
            import matplotlib.pyplot as plt
            plt.imshow(white)
            plt.show()
    
            import matplotlib.pyplot as plt
            plt.imshow(black)
            plt.show()
            
            a = image_tools.path2im('/ipi/research/lameeus/data/hsi/hsi_rgb_grey.png')[..., 0:3]
            # a = (a-0.5)*0.5 + 0.5
            a[red] = [1, 0, 0]
            a[green] = [0, 1, 0]
            a[blue] = [0, 0, 1]
            a[cyan] = [0, 1, 1]
            a[yellow] = [1, 1, 0]
            a[magenta] = [1, 0, 1]
            
            if 0:
                image_tools.save_im(a, '/ipi/research/lameeus/data/hsi/annot2.png')

        img_annot[red, 0] = 1
        img_annot[green, 1] = 1
        img_annot[blue, 2] = 1
        img_annot[cyan, 3] = 1
        img_annot[yellow, 4] = 1
        img_annot[magenta, 5] = 1
        img_annot[white, 6] = 1
        img_annot[black, 7] = 1
        
        np.save(folder_save + 'y_annot.npy', img_annot)
        
    else:
        img_annot = np.load(folder_save + 'y_annot.npy')
        
    return img_annot
    
    
def hsi_annot_test():
    folder_save = '/home/lameeus/data/hsi/load/'

    if 0:  # TODO SET AT 0 after done
        folder_annot = '/ipi/research/lameeus/data/hsi/'
    
        file = folder_annot + 'annot_test.png'
        img = image_tools.path2im(file)[..., 0:3]
    
        shape = np.shape(img)
        shape_annot = [shape[0], shape[1], 8]
        img_annot = np.zeros(shape=shape_annot)
    
        r0 = np.equal(img[:, :, 0], 0)
        r1 = np.equal(img[:, :, 0], 1)
        g0 = np.equal(img[:, :, 1], 0)
        g1 = np.equal(img[:, :, 1], 1)
        b0 = np.equal(img[:, :, 2], 0)
        b1 = np.equal(img[:, :, 2], 1)
    
        red = np.logical_and(np.logical_and(r1, g0), b0)
        green = np.logical_and(np.logical_and(r0, g1), b0)
        blue = np.logical_and(np.logical_and(r0, g0), b1)
        cyan = np.logical_and(np.logical_and(r0, g1), b1)
        yellow = np.logical_and(np.logical_and(r1, g1), b0)
        magenta = np.logical_and(np.logical_and(r1, g0), b1)
        white = np.logical_and(np.logical_and(r1, g1), b1)
        black = np.logical_and(np.logical_and(r0, g0), b0)
    
        if 0:
            import matplotlib.pyplot as plt
            a = image_tools.path2im('/ipi/research/lameeus/data/hsi/hsi_rgb_grey.png')[..., 0:3]
            # a = (a-0.5)*0.5 + 0.5
            a[red] = [1, 0, 0]
            a[green] = [0, 1, 0]
            a[blue] = [0, 0, 1]
            a[cyan] = [0, 1, 1]
            a[yellow] = [1, 1, 0]
            a[magenta] = [1, 0, 1]
            a[white] = [1, 1, 1]
            a[black] = [0, 0, 0]
            
            plt.imshow(a)
            plt.show()
            
        img_annot[red, 0] = 1
        img_annot[green, 1] = 1
        img_annot[blue, 2] = 1
        img_annot[cyan, 3] = 1
        img_annot[yellow, 4] = 1
        img_annot[magenta, 5] = 1
        img_annot[white, 6] = 1
        img_annot[black, 7] = 1
    
        np.save(folder_save + 'y_annot_test.npy', img_annot)
        
    else:
        img_annot = np.load(folder_save + 'y_annot_test.npy')
    
    return img_annot


def hsi_mask():
    
    bool_mask1 = False
    
    if 1:
        from link_to_soliton.paint_tools import image_tools
        
        if bool_mask1:
            mask_img = image_tools.path2im('/home/lameeus/data/hsi/mask.png', type='png')
        else:
            mask_img = image_tools.path2im('/home/lameeus/data/hsi/mask_painting.png')
    
        shape = np.shape(mask_img)
        mask = np.zeros(shape = (shape[0], shape[1]), dtype=int)

        if bool_mask1:
            mask[mask_img[:,:,0] < 0.5] = 1
        else:
            mask[mask_img[:,:,0] > 0] = 1   # just not 0
        
        return mask
