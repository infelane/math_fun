if 1:
    from link_to_soliton.paint_tools import image_tools
    from f2017_08.hsi import tools_plot
    import matplotlib.pyplot as plt
    import numpy as np
    
    folder = '/home/lameeus/Desktop/'
    
    img_ir = image_tools.path2im(folder + 'screen_ir.png')
    img_rgb = image_tools.path2im(folder + 'screen_rgb.png')
    
    h0 = 600+40
    h1 = 800-50
    w0 = 750+50
    w1 = 1000-50
    
    def crop(img, ir = False):
        print(np.shape(img))
        if ir:
            return img[h0:h1, w0:w1, 0:1]
        else:
            return img[h0:h1,w0:w1, 0:3]
    
    def r(img, i = 0):
        print(np.shape(img))
        return img[:,:,i]
    
    img_ir = crop(img_ir, ir = True)
    img_rgb = crop(img_rgb)
    
    plt.figure()
    plt.imshow(img_rgb)
    
    plt.figure()
    for i in range(3):
        img_ir_i = r(img_ir, 0)
        img_rgb_i = r(img_rgb, i)
        
        plt.subplot(3, 3, 3*i+1)
        plt.imshow(img_ir_i, cmap = 'gray')
        plt.subplot(3, 3, 3*i+2)
        plt.imshow(img_rgb_i, cmap = 'gray')
        plt.subplot(3, 3, 3 * i + 3)
        
        mean_rgb = np.mean(img_rgb_i)
        mean_ir = np.mean(img_ir_i)
        std_rgb = np.std(img_rgb_i)
        std_ir = np.std(img_ir_i)

        img_rgb_norm = (img_rgb_i-mean_rgb)/std_rgb
        img_ir_norm = (img_ir_i - mean_ir) / std_ir
        
        plt.imshow(img_rgb_norm-img_ir_norm, cmap='gray', vmin = -0.5, vmax = 0.5)

    mean_rgb = np.mean(img_rgb, axis = (0, 1))
    std_rgb = np.std(img_rgb, axis=(0, 1))
    print(np.shape(mean_rgb))
    img_rgb_norm = (img_rgb - mean_rgb)/std_rgb
    
    mean_ir = np.mean(img_ir, axis = (0, 1))
    std_ir = np.std(img_ir, axis=(0, 1))
    img_ir_norm = (img_ir - mean_ir)/std_ir

    norm_v2 = img_rgb_norm - img_ir_norm

    from sklearn.cluster import KMeans

    shape = np.shape(img_rgb)
    X = np.reshape(norm_v2, (-1, shape[-1]))
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=1).fit(X)
    y = kmeans.labels_
    new_shape = [shape_i for shape_i in shape]
    new_shape[-1] = 1
    y = np.reshape(y, new_shape)

    print(np.shape(y))

    z = tools_plot.n_to_rgb(y[:,:,0], bool_argmax=False)#, anno_col= True)
    
    plt.figure()
    # plt.imshow(y[:,:,0])
    plt.imshow(z)

    plt.show()
    

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
    
if 0:
    import matplotlib.pyplot as plt
    import numpy as np
    np.random.seed(123)
    
    n = 100
    
    x = np.arange(-n//2, n//2)
    
    y_rand = np.random.rand(n) - 0.5
    
    print(y_rand)
    
    folder = '/scratch/Downloads_local/'
    
    def wavelet(x):
        return np.cos(x)*np.exp(-(x)**2/4)
    
    def sigma(x):
        return np.greater_equal(x, 0).astype(float)
    
    # y = np.cos(x/2)*np.exp(-(x/2)**2/2)
    y = wavelet(x/2)
    
    print(np.dot(y_rand, y))
    fig = plt.figure()
    plt.plot(x, y, 'red')
    # fig.savefig(folder + 'sinhan1.png')
    
    conv = np.convolve(y_rand, y)
    fig = plt.figure()
    plt.plot(conv[n // 2:n // 2 + n])
    fig.savefig(folder + 'conv1.png')

    out1 = sigma(conv[n // 2:n // 2 + n])
    fig = plt.figure()
    plt.plot(out1)

    axes = plt.gca()
    axes.set_ylim([-0.5, 1.5])
    fig.savefig(folder + 'out1.png')
    
    # y = np.sin(x/4)
    y = wavelet(x / 4)
    print(np.dot(y_rand, y))
    fig = plt.figure()
    plt.plot(x, y, 'red')
    # fig.savefig(folder + 'sinhan2.png')
    
    conv = np.convolve(y_rand, y)
    fig = plt.figure()
    plt.plot(conv[n//2:n//2+n])
    fig.savefig(folder + 'conv2.png')

    out2 = sigma(conv[n//2:n//2+n])
    fig = plt.figure()
    plt.plot(out2)

    axes = plt.gca()
    axes.set_ylim([-0.5, 1.5])
    fig.savefig(folder + 'out2.png')
    
    
    
    fig = plt.figure()
    plt.plot(x, y_rand, 'blue')
    # fig.savefig(folder + 'temp.png')
    plt.show()
    
    
