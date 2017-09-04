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


class ImgExt(object):
    def __init__(self, img, ext):
        assert ext >= 0
        shape = np.shape(img)
        shape_ext = [shape[0] + 2*ext, shape[1] + 2*ext, shape[2]]
        # TODO something at borders, now greyvalue!
        self.__img_ext = np.ones(shape_ext)*0.5
        self.ext = ext
        if ext > 0:
            self.__img_ext[ext:-ext, ext:-ext, :] = img
        else:
            self.__img_ext[...] = img
    
    def get_crop(self, i_h, i_w, w):
        ext = self.ext
        return self()[i_h: i_h + w +2*ext, i_w: i_w + w + 2*ext, :]
    
    def __call__(self, *args, **kwargs):
        return self.__img_ext


class Data(object):
    w = 10
    def __init__(self, img):
        self.shape = np.shape(img)
        
    def img_to_x(self, img, ext = 0):
    
        w = self.w
        
        n_h = self.shape[0] // w
        n_w = self.shape[1] // w

        self.n_h = n_h
        self.n_w = n_w
        
        shape = (n_h*n_w, w + 2*ext, w + 2*ext, self.shape[2])
        x = np.empty(shape = shape)

        img_ext = ImgExt(img, ext = ext)
        
        for i_h in range(n_h):
            for i_w in range(n_w):
                # x[i_h * n_w + i_w, :,:,:] = img[i_h*self.w:(i_h+1)*w, i_w*w:(i_w+1)*w, :]
                
                
                x[i_h * n_w + i_w, :, :, :] = img_ext.get_crop(i_h * self.w, i_w * self.w, self.w)#[:(i_h + 1) * w, i_w * w:(i_w + 1) * w, :]

        # shape = (self.shape[0]*self.shape[1], 1, 1, self.shape[2])
        # return np.reshape(img, newshape=shape)
        return x
    
    def img_mask_to_x(self, img, mask, ext = 0):
        # TODO adjust the mask according to the width!
        """ img can be a list of img's """
        
        if type(img) == list:
            n_img = len(img)
            
        else:
            n_img = 1
            img = [img]
            ext = [ext]

        img_ext = []
        for i in range(n_img):
            img_ext.append(ImgExt(img[i], ext[i]))
        
        shape = np.shape(img[0])
        folder_xyz =  '/home/lameeus/data/hsi/'
        path_xyz = '/home/lameeus/data/hsi/x_mask.npz'

        x_list = []
        if 0:
            path_coords = '/home/lameeus/data/hsi/coords.npy'
            if 1:
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
            
            n = min(10000, len(coords))
            np.random.seed(123)
            np.random.shuffle(idx)
    
            idx_n = idx[0:n]
    
            coords_n = coords[idx_n, :]
    
            # x = img[coords_n, :]
    
            w = 10
            left = (w - 1) // 2
            right = w - left
            
            # x = [[]]*n_img
            # x = []
            
            # x = []*n_img
            
            # x = {}
            #
            # for i in range(n_img):
            #     x.update({i : []})

            
                # x_i = []
                
            # ext_i = ext[i]
            
            for i_img in range(n_img):
    
                x_i = []
                for i_co in coords_n:
                    h_0 = i_co[0]-left#-ext_i
                    h_1 = i_co[0]+right#+ext_i
                    w_0 = i_co[1]-left#-ext_i
                    w_1 = i_co[1]+right#+ext_i
                    
                   
                    
                    # if h_0 >= 0 & h_1 <= shape[0] & w_0 >= 0 & w_1 <= shape[1]:
                    if (h_0 >= 0) and (h_1 <= shape[0]) and (w_0 >= 0) and (w_1 <= shape[1]):
                        
                        x_i.append(img_ext[i_img].get_crop(h_0, w_0, w))
                        # x[i].append(img[i][h_0:h_1, w_0:w_1, :])
                        # x_i.append(img[i][h_0:h_1, w_0:w_1, :])

                x_i_stack = np.stack(x_i, axis=0)
                np.save(folder_xyz + 'x_mask{}.npy'.format(i_img), x_i_stack)
                
                x_list.append(x_i_stack)
                # for x_i in range(977):
                #     if np.shape(x[x_i])[0] != 10 or np.shape(x[x_i])[1] != 10:
                #         print(np.shape(x_i))
            
            # x_list = []
            # for i in range(n_img):
                # x_list.append(np.stack(x[i], axis=0))
                
                # x.append(x_i)
            
            # np.save(path_xyz, x_list)
            
        else:
            for i_img in range(n_img):
                x_i_stack = np.load(folder_xyz + 'x_mask{}.npy'.format(i_img))
                x_list.append(x_i_stack)
            # x_list = np.load(path_xyz)
            
        if n_img == 1:
            return x_list[0]
        
        else:
            return x_list
  
    def y_to_img(self, y, ext = 0):
        w = self.w
        n_h = self.n_h
        n_w = self.n_w
        
        shape = (self.shape[0], self.shape[1], np.shape(y)[3])
    
        img_y = np.empty(shape = shape)
    
        if ext == 0:
            for i_h in range(n_h):
                for i_w in range(n_w):
                    img_y[i_h * w:(i_h + 1) * w, i_w * w:(i_w + 1) * w, :] = y[i_h * n_w + i_w, :,:,:]
        
        else:
            for i_h in range(n_h):
                for i_w in range(n_w):
                    img_y[i_h * w:(i_h + 1) * w, i_w * w:(i_w + 1) * w, :] = y[i_h * n_w + i_w, ext:-ext, ext:-ext, :]

        return img_y
