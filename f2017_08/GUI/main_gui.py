# # from PIL import ImageTk, Image
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import numpy as np
import matplotlib.backends.backend_tkagg as tkagg

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import Entry, Button, OptionMenu

from link_to_soliton.paint_tools import image_tools

from f2017_08 import registration


def get_list():
    # path = '/scratch/lameeus/data/ghent_altar/input/19_clean.tif'
    # a = {'name': 'clean',
    #      'path': path,
    #      'type': 'ir'}
    
    folder = '/scratch/lameeus/data/ghent_altar/input/'
    
    a = {'name':'clean',
          'path':folder + '19_clean_crop_scale.tif'}
    
    b = {'name':''}
    
    path2 = '/scratch/lameeus/data/ghent_altar/input/19_ir.tif'
    c = {'name': 'ir',
         'path': path2}
        
    d = {'name': 'annotation',
         'path': folder + '19_annot.tif'}
    
    return [a, b, c, d]
    

class PanelImages(object):
    def __init__(self, master, n = 1):
        self.fig = Figure(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(fill = tk.BOTH, expand = 1)
        tkagg.NavigationToolbar2TkAgg(self.canvas, master)
        
        self.a = []
        for i in range(n):
            self.a.append(self.fig.add_subplot(1, n, i+1))

    def get_a(self, i):
        return self.a[i]
    
    def update(self):
        self.canvas.draw()
        
    def set_image(self, im, i = 0):
        self.a[i].imshow(im, interpolation='nearest')
        
    def clear_image(self, i = 0):
        self.a[i].clear

    
class PlotPanel(object):
    def __init__(self, master):
        self.panel_image = PanelImages(master, 2)
        
        # self.fig = Figure(figsize=(6, 6))
        # self.a1 = self.fig.add_subplot(1, 2, 1)
        # self.a2 = self.fig.add_subplot(1, 2, 2)
        # self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        # self.canvas.get_tk_widget().pack()
        # tkagg.NavigationToolbar2TkAgg(self.canvas, master)
        
        self.panel_image.get_a(0).set_title("Estimation Grid", fontsize=16)
        self.panel_image.get_a(0).set_ylabel("Y", fontsize=14)
        self.panel_image.get_a(0).set_xlabel("X", fontsize=14)
    
    def set_image(self, im):
        self.panel_image.get_a(0).clear()
        self.panel_image.get_a(1).clear()

        self.panel_image.get_a(0).imshow(im)
        self.panel_image.get_a(1).imshow(im)

        self.panel_image.update()
        
    def set_overlay(self, im):
        # TODO
        self.panel_image.get_a(0).imshow(im)
        
        
class ButtonAnnot(object):
    def __init__(self, master):
        self.var = tk.IntVar()
        self.button = tk.Checkbutton(master, text='show annotations', variable=self.var)
        self.button.pack()
        
    def get_state(self):
        return self.var.get()
        
    def set_click(self, func):
        self.button.bind('<Button-1>', lambda event: func(self.get_state()))
        # self.button.add
        
        
class FrameRegistration():
    
    def __init__(self, master):
        self.master = master
        
        self.var_ver = tk.IntVar()
        self.var_hor = tk.IntVar()
        self.var_zoom = tk.DoubleVar()

        self.var_ver.set(18)
        self.var_hor.set(2)
        self.var_zoom.set(2.73747126457)
        
        entry_ver = Entry(self.master, textvariable = self.var_ver)
        entry_hor = Entry(self.master, textvariable = self.var_hor)
        entry_zoom = Entry(self.master, textvariable = self.var_zoom)

        entry_ver.pack()
        entry_hor.pack()
        entry_zoom.pack()
        
        self.button = Button(master, text='adjust')
        self.button.pack()
        self.button.bind('<Button-1>', lambda event: self.mix())
        
        folder = '/scratch/lameeus/data/ghent_altar/input/'
        self.im1 = image_tools.path2im(folder + '19_clean.tif')
        self.im2 = image_tools.path2im(folder + '19_rgb.tif')[:, :, 0:3]

        self.panel_image = PanelImages(master, 3)
        self.panel_image.set_image(self.im1, 0)
        self.panel_image.set_image(self.im2, 1)

        self.mix()
        
    # def adjust(self):
    #     print('TODO: SHOULD ADJUST')
        
    def mix(self, style = 'color_mix'):
        if style == 'color_mix':
            delta_h = self.var_ver.get()
            delta_w = self.var_hor.get()
            rescale = self.var_zoom.get()
            
            print(rescale)

            im2_reshape = registration.apply_change(self.im2, rescale)

            im_mix = registration.overlay1(self.im1, im2_reshape, delta_h, delta_w)
            
            self.panel_image.set_image(im_mix, 2)
            self.panel_image.update()
    
            
class Frame1():
    def __init__(self, master):
        """ loading """
        self.window = master
        self.box = Entry(master)
        self.button = Button (master, text="load", command= self.plot)
        self.box.pack ()
        self.button.pack()

        self.dict1 = get_list()

        # l = ['one', '2']

        # self.dict1 = {'test11': path, 'ir': path2}

        self.list1 = [a['name'] for a in self.dict1]

        print(self.dict1[1])

        self.list_variable = tk.StringVar(master)
        self.list_variable.set(self.list1[0])

        # self.optionMenu = OptionMenu(window, variable, l)
        self.optionMenu = OptionMenu(master, self.list_variable, *self.list1)
        self.optionMenu.pack()

        self.plot_panel = PlotPanel(self.window)

        def func(state):
            print('test asdfasdfasdf' )
            print(state)

            list_i = self.list1.index('annotation')
            dict_i = self.dict1[list_i]
            path_i = dict_i['path']
            im = image_tools.path2im(path_i)

            print(im)

            shape_im = np.shape(im)
            im = np.concatenate([im, np.zeros((shape_im[0], shape_im[1], 1))], axis = 2)

            im[im[:, :, 1] == 0, 3] = 1  # is zero for red and green
            # im[im[:,:,0:3] == [1, 0, 0], 3] = 1   # is zero for red and green
            # im[im[:, :, 0:3] == [0, 0, 1], 3] = 1  # is zero for red and green

            self.plot_panel.set_overlay(im)

        self.check_button = ButtonAnnot(master)
        self.check_button.set_click(func)

        # self.fig = Figure(figsize=(6, 6))
        # self.a = self.fig.add_subplot(111)
        # self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        # self.canvas.get_tk_widget().pack()

        """ pre-running some things"""

        self.button.invoke()

    def plot(self):
        list_name = self.list_variable.get()
        list_i = self.list1.index(list_name)
    
        print(list_i)
    
        dict_i = self.dict1[list_i]
    
        path_i = dict_i['path']
        im = image_tools.path2im(path_i)
        # self.a.imshow(im)
    
        # self.a.set_title ("Estimation Grid", fontsize=16)
        # self.a.set_ylabel("Y", fontsize=14)
        # self.a.set_xlabel("X", fontsize=14)
    
        self.plot_panel.set_image(im)
    
        # self.canvas.draw()
    
class TopWindow:
    def __init__(self,  window):
        
        # self.frame = Frame(window)
        # self.frame.pack()
        
        n = ttk.Notebook(window)
        f1 = ttk.Frame(n)  # first page, which would get widgets gridded into it
        f2 = ttk.Frame(n)  # second page
        n.add(f1, text='One')
        n.add(f2, text='Two')
        
        n.pack(fill = tk.BOTH)

        Frame1(f2)
        FrameRegistration(f1)
        

def main():
    window = tk.Tk()
    start = TopWindow(window)
    window.mainloop()
    

if __name__ == '__main__':
    main()
    