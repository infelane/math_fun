""" Tools for processing data
"""

import numpy as np


def norm_hsi(a):
    """ normalization specific to the hsi dataset
    """

    if 0:
        img_max = np.max(a)
        img_min = np.min(a)
        print(img_max)
        print(img_min)

    a_norm = np.empty(shape=np.shape(a))
    a_norm[...] = (a + 100) / 2046.

    # show_histo(img)
    # show_histo(capped(img_norm))

    return capped01(a_norm)


def capped01(a):
    a[a > 1.] = 1.
    a[a < 0.] = 0.
    return a


def norm01_to_norm11(a):
    """ from [0, 1] to [-1, 1]
    """
    
    return a*2 - 1


def norm11_to_norm01(a):
    """ from [0, 1] to [-1, 1]
    """
    
    return (a+1)*0.5


class Data(object):
    w = 10
    def __init__(self, img):
        self.shape = np.shape(img)
        
    def img_to_x(self, img):
    
        w = self.w
        
        n_h = self.shape[0] // w
        n_w = self.shape[1] // w

        
        self.n_h = n_h
        self.n_w = n_w
        
        shape = (n_h*n_w, w, w, self.shape[2])
        x = np.empty(shape = shape)
        
        for i_h in range(n_h):
            for i_w in range(n_w):
                x[i_h * n_w + i_w, :,:,:] = img[i_h*self.w:(i_h+1)*w, i_w*w:(i_w+1)*w, :]
        
        # shape = (self.shape[0]*self.shape[1], 1, 1, self.shape[2])
        # return np.reshape(img, newshape=shape)
        return x
    
    def img_mask_to_x(self, img, mask):
        # TODO adjust the mask according to the width!
        
        shape = np.shape(img)
        path_x = '/home/lameeus/data/hsi/x_mask.npy'
        
        if 0:
            
            path_coords = '/home/lameeus/data/hsi/coords.npy'
            if 0:
                coords = []
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        if mask[i, j] == 1:
                            coords.append([i, j])
        
                coords = np.array(coords)
                np.save(path_coords, coords)
    
            else:
                coords = np.load(path_coords)
    
            print(np.shape(coords))
            idx = np.arange(np.shape(coords)[0])
            n = 10000
            np.random.shuffle(idx)
    
            idx_n = idx[0:n]
    
            coords_n = coords[idx_n, :]
    
            # x = img[coords_n, :]
    
            w = 10
            left = (w - 1) // 2
            right = w - left
            ext = 0
            
            x = []
            for i in coords_n:
                h_0 = i[0]-left-ext
                h_1 = i[0]+right+ext
                w_0 = i[1]-left-ext
                w_1 = i[1]+right+ext
                
                # if h_0 >= 0 & h_1 <= shape[0] & w_0 >= 0 & w_1 <= shape[1]:
                if (h_0 >= 0) and (h_1 <= shape[0]) and (w_0 >= 0) and (w_1 <= shape[1]):
                    x.append(img[h_0:h_1, w_0:w_1, :])

            # for x_i in range(977):
            #     if np.shape(x[x_i])[0] != 10 or np.shape(x[x_i])[1] != 10:
            #         print(np.shape(x_i))
                    
            x = np.stack(x, axis=0)
            np.save(path_x, x)
            
        else:
            x = np.load(path_x)
            
        return x
  
    def y_to_img(self, y):
        w = self.w
        n_h = self.n_h
        n_w = self.n_w
        
        shape = (self.shape[0], self.shape[1], np.shape(y)[3])
    
        img_y = np.empty(shape = shape)
    
        for i_h in range(n_h):
            for i_w in range(n_w):
                img_y[i_h * w:(i_h + 1) * w, i_w * w:(i_w + 1) * w, :] = y[i_h * n_w + i_w, :,:,:]
    
        return img_y
