""" Tools for processing data
"""

import numpy as np

from f2017_08.hsi import tools_plot


def norm_hsi(img):
    """ normalization specific to the hsi dataset
    """

    if 0:
        img_max = np.max(img)
        img_min = np.min(img)
        print(img_max)
        print(img_min)

    a_norm = np.empty(shape=np.shape(img))
    a_norm[...] = (img + 100) / 2046.

    a_capped = capped01(a_norm)


    if 1:
        tools_plot.show_histo(img, show = False)
        tools_plot.show_histo(a_capped)

    return a_capped

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
        shape = np.shape(img)
        if type(ext) is tuple:
            assert len(ext) == 2
            assert ext[0] >= 0
            assert ext[1] >= 0
            shape_ext = [shape[0] + ext[0] + ext[1], shape[1] + ext[0] + ext[1], shape[2]]
        
        elif type(ext) is int:
            assert ext >= 0
            shape_ext = [shape[0] + 2 * ext, shape[1] + 2 * ext, shape[2]]
            
        else:
            raise TypeError
            
        # TODO something at borders, now greyvalue!
        self.__img_ext = np.ones(shape_ext)*0.5
        self.ext = ext

        if type(ext) is tuple:
            self.__img_ext[ext[0]:ext[0]+shape[0], ext[0]:ext[0]+shape[1], :] = img

        elif type(ext) is int:
            
            if ext > 0:
                self.__img_ext[ext:-ext, ext:-ext, :] = img
            else:
                self.__img_ext[...] = img
    
    def get_crop(self, i_h, i_w, w):
        ext = self.ext
        if type(ext) is tuple:
            return self()[i_h: i_h + w + ext[0] + ext[1], i_w: i_w + w + ext[0] + ext[1], :]
        elif type(ext) is int:
            return self()[i_h: i_h + w + 2*ext, i_w: i_w + w + 2*ext, :]
    
    def __call__(self, *args, **kwargs):
        return self.__img_ext


class Data(object):
    def __init__(self, img, w = 10):
        self.shape = np.shape(img)
        self.w = w
        
    def img_to_x(self, img: object, ext: object = 0) -> object:
    
        w = self.w
        
        shape_in = np.shape(img)
        
        n_h = self.shape[0] // w
        n_w = self.shape[1] // w

        self.n_h = n_h
        self.n_w = n_w
        
        if type(ext) is tuple:
            assert len(ext) == 2
            shape = (n_h * n_w, w + ext[0] + ext[1], w + ext[0] + ext[1], shape_in[2])
        
        elif type(ext) is int:
            shape = (n_h * n_w, w + 2 * ext, w + 2 * ext, shape_in[2])
        
        else:
            raise TypeError('ext is expected to be tuple or integer')
        
        
        x = np.empty(shape = shape)

        img_ext = ImgExt(img, ext = ext)
        
        for i_h in range(n_h):
            for i_w in range(n_w):
                x[i_h * n_w + i_w, :, :, :] = img_ext.get_crop(i_h * self.w, i_w * self.w, self.w)#[:(i_h + 1) * w, i_w * w:(i_w + 1) * w, :]

        return x
    
    def img_mask_to_x(self, img, mask, w = 10, ext = 0, name = None, bool_new = False):
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

        

        if name is None:
            file_name = 'x_mask{}.npy'
        else:
            file_name = 'x_' + name + '_mask{}.npy'

        try:
            if bool_new:
                raise PermissionError('New data is wanted')
            x_list = []
            for i_img in range(n_img):
                x_i_stack = np.load(folder_xyz + file_name.format(i_img))
                x_list.append(x_i_stack)

        except:
            x_list = []
            if name:
                path_coords = folder_xyz + 'coords_' + name + '.npy'
            else:
                path_coords = folder_xyz + 'coords.npy'
    
            try:
                if bool_new:
                    raise PermissionError('New coords data is wanted')
                coords = np.load(path_coords)
                
            except:
                coords = []
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        if mask[i, j] == 1:
                            coords.append([i, j])
    
                coords = np.array(coords)
                np.save(path_coords, coords)
                
    
            print(np.shape(coords))
            idx = np.arange(np.shape(coords)[0])
            
            n = min(10000, len(coords))
            np.random.seed(123)
            np.random.shuffle(idx)
    
            idx_n = idx[0:n]
    
            coords_n = coords[idx_n, :]
        
            left = (w - 1) // 2
            right = w - left
            
            for i_img in range(n_img):
    
                x_i = []
                for i_co in coords_n:
                    h_0 = i_co[0]-left
                    h_1 = i_co[0]+right
                    w_0 = i_co[1]-left
                    w_1 = i_co[1]+right
                    
                    if (h_0 >= 0) and (h_1 <= shape[0]) and (w_0 >= 0) and (w_1 <= shape[1]):
                        x_i.append(img_ext[i_img].get_crop(h_0, w_0, w))

                x_i_stack = np.stack(x_i, axis=0)
                
                np.save(folder_xyz + file_name.format(i_img), x_i_stack)
                
                x_list.append(x_i_stack)
                
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
