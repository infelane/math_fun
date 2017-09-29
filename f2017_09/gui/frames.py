from tkinter import Frame, Canvas
# from tkinter.ttk import
from PIL import Image, ImageTk
import numpy as np


class PanelImages3(object):
    def __init__(self, master):
        """ with rescaling """
        
        # TODO when resizing window, rescale canvas
        # TODO adjust canvas size to image (keep image in center)

        #
        #     self.w = w
        #
        #     self.zoomcycle = 0
        #     self.zimg_id = None
        #
        #     master.bind("<MouseWheel>", self.zoomer)
        #     self.w.bind("<Motion>", self.crop)
        
        self.canvas = Canvas(master, width = 100, height = 100)
        self.canvas.pack(fill='both', expand=1)
        
        self.orig_img = None
        self.img = None
        self.canvas.create_image(0, 0, image=None, anchor="nw")
        
        self.zoomcycle = 0
        self.zimg_id = None
        
        self.canvas.bind("<Motion>", self.crop)
        self.canvas.bind("<Button-4>", self.zoomer)
        self.canvas.bind("<Button-5>", self.zoomer)
    
    def zoomer(self, event):
        if (event.delta > 0):
            if self.zoomcycle != 4: self.zoomcycle += 1
        elif (event.delta < 0):
            if self.zoomcycle != 0: self.zoomcycle -= 1
        
        if event.num == 4:
            if self.zoomcycle != 4: self.zoomcycle += 1
        elif event.num == 5:
            if self.zoomcycle != 0: self.zoomcycle -= 1
        
        self.crop(event)
    
    def crop(self, event):
        if self.zimg_id: self.canvas.delete(self.zimg_id)
        if (self.zoomcycle) != 0:
            x, y = event.x, event.y
            
            x_rel = x/self.new_shape[1]
            y_rel = y/self.new_shape[0]
            
            x_orig = x_rel * self.shape_orig[1]
            y_orig = y_rel * self.shape_orig[0]
            
            if self.zoomcycle == 1:
                w_half_crop = 45
                h_half_crop = 30
            elif self.zoomcycle == 2:
                w_half_crop = 30
                h_half_crop = 20
            elif self.zoomcycle == 3:
                w_half_crop = 15
                h_half_crop = 10
            elif self.zoomcycle == 4:
                w_half_crop = 6
                h_half_crop = 4

            tmp = self.orig_img.crop((x_orig - w_half_crop, y_orig - h_half_crop,
                                      x_orig + w_half_crop, y_orig + h_half_crop))
                
            size = 300, 200
            self.zimg = ImageTk.PhotoImage(tmp.resize(size))
            self.zimg_id = self.canvas.create_image(event.x, event.y, image=self.zimg)
            
            #     self.fig = Figure(figsize=(6, 6))
            #     self.canvas = FigureCanvasTkAgg(self.fig, master=master)
            #     self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
            #     tkagg.NavigationToolbar2TkAgg(self.canvas, master)
            #
            #     self.a = []
            #     for i in range(n):
            #         self.a.append(self.fig.add_subplot(1, n, i + 1))
            #
            # def get_a(self, i):
            #     return self.a[i]
            #
    
    def set_image(self, im):
        shape = (self.canvas.winfo_height(), self.canvas.winfo_width())
        
        if np.max(im) <= 1:
            im = (im*255).astype(np.uint8)
        elif np.max(im) <= 255:
            im = im.astype(np.uint8)
        
        img = Image.fromarray(im, 'RGB')
        self.orig_img = img
        
        self.shape_orig = (img.height, img.width )   # to keep height, width order
        
        ratio_canvas = shape[0] / shape[1]  # Height vs width
        ratio_image = self.shape_orig[0] / self.shape_orig[1]
        
        if ratio_image >= ratio_canvas:
            # image will be put on LEFT of canvas
            new_shape = (shape[0], int(np.ceil(shape[0] / ratio_image)))
        else:
            # image will be on top of canvas
            new_shape = (int(np.ceil(shape[1] * ratio_image)), shape[1])
        
        self.new_shape = new_shape
        img = img.resize((new_shape[1], new_shape[0]), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, image=self.img, anchor="nw")
        
        self.canvas.update_idletasks()