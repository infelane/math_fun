if 0:
    from link_to_soliton.paint_tools import image_tools
    from matplotlib import pyplot as plt
    import numpy as np
    
    folder = '/home/lameeus/data/ghent_altar/output/'

    image_annot = image_tools.path2im(folder + 'hand_small.tif')

    r1 = np.equal(image_annot[:, :, 0], 1)
    b1 = np.equal(image_annot[:, :, 2], 1)
    g0 = np.equal(image_annot[:, :, 1], 0)
    b0 = np.equal(image_annot[:, :, 2], 0)

    red = r1 * b0 * g0

    image_annot2 = np.copy(image_annot)
    image_annot2[red, :] = [0, 1, 1]
    
    # folder_out = '/home/lameeus/data/ghent_altar/restorers_inpainting/'
    #
    # image_tools.save_im(image_annot2, folder + 'hand_small_cyan.tif')
    
    plt.imshow(image_annot2)
    plt.show()
    

if 0:
    from link_to_soliton.paint_tools import image_tools
    from matplotlib import pyplot as plt
    import numpy as np
    
    # Crop
    folder = '/home/lameeus/data/ghent_altar/bart_devolder/02/'
    
    image = image_tools.path2im(folder + 'x103112l.tif')
    
    if 0:
        h0 = 2724
        h1 = 3446
        
        w0 = 2650
        w1 = 3728
    else:
        h0 = 1032
        h1 = 1918
    
        w0 = 3881
        w1 = 4549
    image_crop = image[h0:h1, w0:w1, ...]
    
    folder_out = '/home/lameeus/data/ghent_altar/restorers_inpainting/'

    image_tools.save_im(image_crop, folder_out + '13_zach_restored.tif')

    plt.imshow(image_crop)
    plt.show()


if 0:
    from link_to_soliton.paint_tools import image_tools
    from matplotlib import pyplot as plt
    import numpy as np

    folder = '/home/lameeus/data/ghent_altar/input/'
    if 0:
        # Crop
      
        image = image_tools.path2im(folder + '19_annot_clean_big.tif')
        
        h0 = 4399
        h1 = h0 + 1401
        
        w0 = 2349
        w1 = w0 + 2101
        image_crop = image[h0:h1, w0:w1, ...]
        
        # folder_out = '/home/lameeus/data/ghent_altar/input/19_hand/'
        
        # image_tools.save_im(image_crop, folder_out + '19_hand_annot_clean.tif')
        
    else:
        image = image_tools.path2im(folder + '13_annot_clean.tif')

        # h0 = 4399
        # h1 = h0 + 1401
        #
        # w0 = 2349
        # w1 = w0 + 2101
        image_crop = image[h0:h1, w0:w1, ...]
        
    
    plt.imshow(image_crop)
    plt.show()
    

if 0:
    from link_to_soliton.paint_tools import image_tools
    from matplotlib import pyplot as plt
    import numpy as np
    
    if 0:
        # Crop
        folder = '/home/lameeus/data/ghent_altar/inpainting/detection_and_inpainting/Zachary/'
    
        image = image_tools.path2im(folder + 'src_i3_with.png')
        image_good = image_tools.path2im(folder + 'cut_ori_3.png')
        
        shape = np.shape(image_good)
        
        h0 = 651
        w0 = 343
        image_crop = image[h0:h0+shape[0],w0:w0+shape[1],:]
    else:
        folder = '/home/lameeus/data/ghent_altar/input/'
        image_big = image_tools.path2im(folder + '13_annot_clean.tif')
        
        folder_small = '/scratch/lameeus/data/ghent_altar/altarpiece_close_up/beard_updated/'
        image_good = image_tools.path2im(folder_small + 'rgb_cleaned.tif')

        shape = np.shape(image_good)

        h0 = 406
        w0 = 3557
        image_crop = image_big[h0:h0 + shape[0], w0:w0 + shape[1], :]
    
    folder_out = '/home/lameeus/data/ghent_altar/annotation/'
    image_tools.save_im(image_crop, folder_out + '13_zach_small_annot__clean.tif')
    
    plt.imshow(image_crop)
    plt.show()
    

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
    
if 1:
    from link_to_soliton.paint_tools import image_tools
    from matplotlib import pyplot as plt
    import numpy as np
    
    folder = '/scratch/lameeus/data/ghent_altar/altarpiece_close_up/beard_updated/'
    image_in = image_tools.path2im(folder + 'rgb_cleaned.tif')
    
    folder_map = '/home/lameeus/data/ghent_altar/annotation/'
    image_map = image_tools.path2im(folder_map + '13_zach_small_annot_clean.tif')

    image_map2 = np.copy(image_in)

    h0 = 150
    h1 = 450
    w0 = 371
    w1 = 671
    image_map2[h0:h1, w0:w1, :] = image_map[h0:h1, w0:w1, :]
    
    h0 = 1292
    h1 = 1692
    w0 = 502
    w1 = 902
    image_map2[h0:h1, w0:w1, :] = image_map[h0:h1, w0:w1, :]

    h0 = 1623
    h1 = 1923
    w0 = 912
    w1 = 1212
    image_map2[h0:h1, w0:w1, :] = image_map[h0:h1, w0:w1, :]

    r1 = np.equal(image_map2[:, :, 0], 1)
    g0 = np.equal(image_map2[:, :, 1], 0)
    g1 = np.equal(image_map2[:, :, 1], 1)
    b0 = np.equal(image_map2[:, :, 2], 0)
    b1 = np.equal(image_map2[:, :, 2], 1)
    
    if 1:
        """ only squared regions """
    
    red_map = r1 * b0 * g0

    image_annot = np.copy(image_in)
    cyan = [0, 1, 1]
    red = [1, 0 , 0]


    image_annot[red_map, :] = red

    
    folder = '/home/lameeus/data/ghent_altar/annotation/'
    image_tools.save_im(image_map2, folder +  '13_zach_small_annot2.tif')
    image_tools.save_im(image_annot, folder + '13_zach_small_annot2_red.tif')

    plt.subplot(1, 2, 1)
    plt.imshow(image_map2)
    plt.subplot(1, 2, 2)
    plt.imshow(image_annot)
    plt.show()
    