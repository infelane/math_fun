

if 0:
    from link_to_soliton.paint_tools import image_tools
    from matplotlib import pyplot as plt
    import numpy as np
    
    folder = '/scratch/lameeus/data/ghent_altar/altarpiece_close_up/finger/'
    image_clean = image_tools.path2im(folder + 'hand_cleaned.tif')

    folder = '/home/lameeus/data/ghent_altar/annotation/'
    image_annot = image_tools.path2im(folder + '19_annot_clean_small.tif')

    image_annot2 = np.copy(image_clean)
    
    r1 = np.equal(image_annot[:,:,0], 1)
    b1 = np.equal(image_annot[:,:,2], 1)
    g0 = np.equal(image_annot[:,:,1], 0)
    b0 = np.equal(image_annot[:, :, 2], 0)
    
    red = r1*b0*g0

    image_annot2[red, :] = [1, 0, 0]
    
    image_tools.save_im(image_annot2, '/home/lameeus/data/ghent_altar/annotation/19_annot_small_clean_red.tif')
    
    plt.imshow(image_annot2)
    plt.show()
