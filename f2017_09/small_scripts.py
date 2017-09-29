if 0:
    from link_to_soliton.paint_tools import image_tools
    import numpy as np

    folder = '/home/lameeus/data/ghent_altar/input/'
    
    img_big = image_tools.path2im(folder + '19_clean.tif')
    img_annot_big = image_tools.path2im(folder + '19_annot_big.tif')
    
    img_annot_big_clean = np.copy(img_big)
    
    bool_map = np.greater(3, np.sum(img_annot_big, axis=-1))  # white background
    img_annot_big_clean[bool_map, :] = img_annot_big[bool_map, :]
    
    image_tools.save_im(img_annot_big_clean, folder + '19_annot_clean_big.tif')
    
    h0 = 4399
    h1 = h0 + 1401
    
    w0 = 2349
    w1 = w0 + 2101
    
    image_tools.save_im(img_annot_big_clean[h0:h1, w0:w1, :],
                        '/home/lameeus/data/ghent_altar/annotation/' + '19_annot_clean_big.tif')
