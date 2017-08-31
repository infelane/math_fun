
import tkinter as tk
from tkinter import ttk, Entry, Button

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn import decomposition

from f2017_08.hsi import tools_data, tools_datasets

"""
Comments:
There are outliers in the data set, if we crop they are removed. I'll just crop for now
"""

def norm_img(img):
    img_max = np.max(img)
    img_min = np.min(img)
    diff = (img_max/2 - img_min/2)  # had problems with overflow
    
    print(img_max)
    print(img_min)
    
    img_norm = ((img - img_min)/diff)/2
    # return capped(img_norm)    # TO float [0, 1], stupid way to avoid overflow
    return img_norm

def norm_img2(img):
    raise LookupError('moved to tools_data/norm_hsi')
    
def capped(x):
    raise LookupError('moved to tools_data/norm_hsi')

def example_crop(img):
    delta = 500
    return img[0:-delta, 0:-delta, :]


class Flat(object):
    def __init__(self, img):
        self.shape = np.shape(img)
    
    def flatten(self, img):
        X = np.reshape(img, (-1, self.shape[2]))
        return X
    
    def deflatten(self, X):
        return np.reshape(X, (self.shape[0], self.shape[1], -1))


def show_histo(img):
    hist, bin_edges = np.histogram(img, bins = 20, density=True)
    bin_center = (bin_edges[0:-1] + bin_edges[1:])/2

    plt.plot(bin_center, hist)  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()
    

def get_i_rgb(wavelengths):
    i_rgb = np.searchsorted(wavelengths, [700, 520, 440], side='right')
    return i_rgb


def norm01(x):
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    x_norm = (x - x_min)/(x_max - x_min)
    
    return x_norm


class PcaStuff(object):
    def __init__(self, img, img_data = None, n = 1, new = False):
        self.flat = Flat(img)
        
        self.pca = decomposition.PCA(n_components=n)
        
        img_flat = self.flat.flatten(img)

        if new:
            if img_data is not None:
                self.pca.fit(img_data)
            else:
                self.pca.fit(img_flat)
                
            self.save()
        else:
            self.pca.fit(img_flat[0:n, :])   # No real fitting, but to enable to load the values
            self.load()
            
        self.x = self.pca.transform(img_flat)
        self.x_norm = norm01(self.x)
        
    def load(self):
        self.pca.components_, self.pca.mean_ = np.load('pca.npy')
        
    def save(self):
        np.save('pca.npy', [self.pca.components_, self.pca.mean_])
        
    def get_x(self):
        
        img_x = self.flat.deflatten(self.x_norm)
        return img_x
    
    def get_img_filtered(self, i = None):
        
        print(i.__class__)
        
        if i == None:
            img_pca_filter_flat = self.pca.inverse_transform(self.x)
        elif i.__class__ == int:
            x_i = np.zeros(shape=np.shape(self.x))
            x_i[:, i] = self.x[:, i]
            img_pca_filter_flat = self.pca.inverse_transform(x_i)
        else:
            x_i = np.zeros(shape=np.shape(self.x))
            
            print(i)
            
            for ii in i:
                
                print(ii)
                
                x_i[:, ii] = self.x[:, ii]
                
            img_pca_filter_flat = self.pca.inverse_transform(x_i)
        
        img_filtered = capped(self.flat.deflatten(img_pca_filter_flat))
        
        return img_filtered


class HsiData(object):
    def __init__(self, load = True):
        if load:
            img = tools_datasets.hsi_processed()
       
            # image3D = tools_data.Image3D()
            # img = image3D.get_img()
            # wavelengths = image3D.get_wavelength()
            
            self.img = tools_data.norm_hsi(img)
            # self.img = norm_img2(img)
    
        self.mean = 0.18 # np.mean(self.img)

        wavelengths = tools_datasets.hsi_raw()['wavelengths']
        self.i_rgb =  get_i_rgb(wavelengths)
        
    def get_img(self):
        return self.img
    
    def get_data(self):
        """ removes the border """
        mask = np.array(Image.open('/ipi/research/lameeus/data/hsi_raw/mask.png'))
        shape = np.shape(mask)
        mask_01 = np.zeros(shape=(shape[0], shape[1]))
        mask_01[mask[:, :, 0] == 0] = 1
        return self.img[mask_01 == 1, :]
    
    # def get_i_rgb(self):
    #     return self.i_rgb
    
    def to_rgb(self, img):
        shape = np.shape(img)
        rgb = np.zeros((shape[0], shape[1], 3))
        
        # rgb[...] = img[:, :, self.i_rgb]
        
        for i in range(3):
            color_i = self.i_rgb[i]
            color_conv = np.arange(color_i-5,color_i+5)
            color_conv = color_conv[color_conv > 0]
            color = np.mean(img[:, :, color_conv], axis=2)  # mean color intensities
            rgb[:, :, i] = color
        
        # r_i = self.i_rgb[0]
        # r_conv = np.arange(r_i-10,r_i+10)
        # r = np.mean(img[:, :, r_conv], axis = 2)    # mean

        # rgb[:,:,0] = r
        
        # print(np.mean(rgb))
        
        return tools_data.capped01(rgb/(self.mean*2)) # to make mean 0.5


def main():
    if 0:
        interesting()

    hsi_data = HsiData()
    img = hsi_data.get_img()
    img_data = hsi_data.get_data()
    
    # img = example_crop(img)

    # i_rgb = hsi_data.get_i_rgb()

    pca_stuff = PcaStuff(img, img_data = img_data, n = 10, new = False)
    img_pca_filter = pca_stuff.get_img_filtered()
    i = 2
    img_pca_filter_i = pca_stuff.get_img_filtered(i = i)
    img_x = pca_stuff.get_x()
    
    plt.subplot(1, 3, 1)
    plt.imshow(hsi_data.to_rgb(img))
    plt.title('original')
    plt.subplot(1, 3, 2)
    plt.imshow(hsi_data.to_rgb(img_pca_filter_i))
    plt.title('pca filtering')
    plt.subplot(1, 3, 3)
    plt.imshow(img_x[..., i])
    plt.title('pca components')
    plt.show()
    
    plt.subplot(1, 3, 1)
    plt.imshow(hsi_data.to_rgb(img))
    plt.title('original')
    plt.subplot(1, 3, 2)
    plt.imshow(hsi_data.to_rgb(img_pca_filter))
    plt.title('pca filtering')
    plt.subplot(1, 3, 3)
    plt.imshow(img_x[..., 0:3])
    plt.title('pca components')
    plt.show()
    
    plt.plot(np.mean(img, axis=(0, 1)), label = 'mean')
    plt.plot(np.min(img, axis=(0, 1)), label = 'min')
    plt.plot(np.max(img, axis=(0, 1)), label = 'max')
    plt.legend()
    plt.show()


def interesting():
    image3D = tools_data.Image3D()
    img = image3D.get_img()
    wavelengths = image3D.get_wavelength()
    img_norm = norm_img2(img)

    i_rgb = get_i_rgb(wavelengths)

    pca_stuff = PcaStuff(img_norm, n = 3)
    img_pca_filter = pca_stuff.get_img_filtered()
    img_x = pca_stuff.get_x()

    plt.subplot(1, 3, 1)
    plt.imshow(img_norm[:, :, i_rgb])
    plt.title('original')
    plt.subplot(1, 3, 2)
    plt.imshow(img_pca_filter[:, :, i_rgb])
    plt.title('pca filtering')
    plt.subplot(1, 3, 3)
    plt.imshow(img_x)
    plt.title('pca components')
    plt.show()


class Frame1():
    def __init__(self, master):
        """ loading """
        self.window = master

        self.button_next = Button(master, text="next") #, command=self.plot)
        self.button_prev = Button(master, text="prev")  # , command=self.plot)
       
        sv = tk.StringVar()

        n = 10
        
        self.lst = [a for a in range(n)]

        sv.set(self.lst)
        
        def callback(sv):
            string = sv.get()
            lst_string = string.split()
            self.lst = [int(a) for a in lst_string]
            
        sv.trace("w", lambda name, index, mode, sv=sv: callback(sv))
        self.entry = Entry(master, textvariable = sv)

        self.button_next.pack()
        self.button_prev.pack()
        self.entry.pack()

        hsi_data = HsiData()
        img = hsi_data.get_img()
        img_data = hsi_data.get_data()

        # img = example_crop(img)

        # i_rgb = hsi_data.get_i_rgb()

        pca_stuff = PcaStuff(img, img_data=img_data, n=n, new = False)
        
        def gen_rgb():
            print(self.lst)
            img_new = pca_stuff.get_img_filtered(i=self.lst)
            
            print(np.shape(img_new))
            
            panel_image.set_image(hsi_data.to_rgb(img_new), i=0)
            
        rgb = hsi_data.to_rgb(img)
        
        # plt.imshow(rgb)
        # plt.show()
        
        from f2017_08.GUI import main_gui
        
        panel_image = main_gui.PanelImages(master, n = 2)
        panel_image.set_image(rgb, i = 0)
        panel_image.clear_image(i = 0)
        
        panel_image.update()
        
        self.i = 0
        
        def update_im():
            gen_rgb()
            
            img_pca_filter_i = pca_stuff.get_img_filtered(i=self.i)
            rgb = hsi_data.to_rgb(img_pca_filter_i)
            panel_image.set_image(rgb, i=1)
            panel_image.clear_image(i=1)
            panel_image.update()
        
        def func_next():
            self.i += 1
            update_im()
            
        def func_prev():
            self.i -= 1
            update_im()

        self.button_next.bind('<Button-1>', lambda event: func_next())
        self.button_prev.bind('<Button-1>', lambda event: func_prev())

    #     self.dict1 = get_list()
    #
    #     # l = ['one', '2']
    #
    #     # self.dict1 = {'test11': path, 'ir': path2}
    #
    #     self.list1 = [a['name'] for a in self.dict1]
    #
    #     print(self.dict1[1])
    #
    #     self.list_variable = tk.StringVar(master)
    #     self.list_variable.set(self.list1[0])
    #
    #     # self.optionMenu = OptionMenu(window, variable, l)
    #     self.optionMenu = OptionMenu(master, self.list_variable, *self.list1)
    #     self.optionMenu.pack()
    #
    #     self.plot_panel = PlotPanel(self.window)
    #
    #     def func(state):
    #         print('test asdfasdfasdf')
    #         print(state)
    #
    #         list_i = self.list1.index('annotation')
    #         dict_i = self.dict1[list_i]
    #         path_i = dict_i['path']
    #         im = image_tools.path2im(path_i)
    #
    #         print(im)
    #
    #         shape_im = np.shape(im)
    #         im = np.concatenate([im, np.zeros((shape_im[0], shape_im[1], 1))], axis=2)
    #
    #         im[im[:, :, 1] == 0, 3] = 1  # is zero for red and green
    #         # im[im[:,:,0:3] == [1, 0, 0], 3] = 1   # is zero for red and green
    #         # im[im[:, :, 0:3] == [0, 0, 1], 3] = 1  # is zero for red and green
    #
    #         self.plot_panel.set_overlay(im)
    #
    #     self.check_button = ButtonAnnot(master)
    #     self.check_button.set_click(func)
    #
    #     # self.fig = Figure(figsize=(6, 6))
    #     # self.a = self.fig.add_subplot(111)
    #     # self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
    #     # self.canvas.get_tk_widget().pack()
    #
    #     """ pre-running some things"""
    #
    #     self.button.invoke()
    #
    # def plot(self):
    #     list_name = self.list_variable.get()
    #     list_i = self.list1.index(list_name)
    #
    #     print(list_i)
    #
    #     dict_i = self.dict1[list_i]
    #
    #     path_i = dict_i['path']
    #     im = image_tools.path2im(path_i)
    #     # self.a.imshow(im)
    #
    #     # self.a.set_title ("Estimation Grid", fontsize=16)
    #     # self.a.set_ylabel("Y", fontsize=14)
    #     # self.a.set_xlabel("X", fontsize=14)
    #
    #     self.plot_panel.set_image(im)
    #
    #     # self.canvas.draw()
    

class Window:
    def __init__(self, window):
        # self.frame = Frame(window)
        # self.frame.pack()
        
        n = ttk.Notebook(window)
        f1 = ttk.Frame(n)  # first page, which would get widgets gridded into it
        f2 = ttk.Frame(n)  # second page
        n.add(f1, text='One')
        n.add(f2, text='Two')
        
        n.pack(fill=tk.BOTH, expand = 1)
        
        Frame1(f1)
        # FrameRegistration(f1)
    

def main_gui():
    window = tk.Tk()
    start = Window(window)
    window.mainloop()


if __name__ == '__main__':
    # main()
    main_gui()
