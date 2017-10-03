from tkinter import Frame, Canvas
# from tkinter.ttk import
from PIL import Image, ImageTk
import numpy as np
from tkinter.ttk import Button


class PanelDualImage(object):
    def __init__(self, master):
        
        f = Frame(master)
        f.pack(side = 'top', fill = 'both', expand = 1)
        
        self.canvas_left = Canvas(f)
        # self.canvas_left.pack(fill='both', expand=1)
        #
        self.canvas_right = Canvas(f)
        # self.canvas_right.pack(fill='both', expand=1)
        
   
            
        f = Frame(master)
        f.pack(side='bottom')
        
        b = Button(f, text='single', command=self.single)
        b.pack(side = 'left', anchor = 'e')
        b = Button(f, text='top-bottom', command=self.topbot)
        b.pack(side='left', anchor='e')
        # b.pack(side = None)
        b = Button(f, text='left-right', command=self.leftright)
        b.pack(side='left', anchor='e')
        # b.pack(side = 'right', anchor = 'w')

        self.start_widgets()

    #     self.orig_img = None
    #     self.img = None
    #     self.canvas.create_image(0, 0, image=None, anchor="nw")

        self.zoomcycle = 0
        self.zimg_id_left = None
        self.zimg_id_right = None
        self.orig_img_left = None
        self.orig_img_right = None

        self.canvas_left.bind("<Motion>", self.crop)
        self.canvas_left.bind("<Button-4>", self.zoomer)
        self.canvas_left.bind("<Button-5>", self.zoomer)
        
        self.canvas_right.bind("<Motion>", self.crop)
        self.canvas_right.bind("<Button-4>", self.zoomer)
        self.canvas_right.bind("<Button-5>", self.zoomer)

        self.canvas_left.bind("<Configure>", lambda event: self.set_image(self.orig_img_left, 0) )
        self.canvas_right.bind("<Configure>", lambda event: self.set_image(self.orig_img_right, 1) )

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
        if self.orig_img_left is None and self.orig_img_right is None: return    # no image yet
        
        if self.zimg_id_left:
            self.canvas_left.delete(self.zimg_id_left)
        if self.zimg_id_right:
            self.canvas_right.delete(self.zimg_id_right)
            
        if (self.zoomcycle) != 0:
            x, y = event.x, event.y

            x_rel = x / self.new_shape[1]
            y_rel = y / self.new_shape[0]

            x_orig = x_rel * self.shape_orig[1]
            y_orig = y_rel * self.shape_orig[0]

            size = 600, 400     # first 300, 200
            
            if self.zoomcycle == 1:
                # x2
                # TODO for test x1
                
                zoom_fac = 2
                
                # w_half_crop = 45
                # h_half_crop = 30
            elif self.zoomcycle == 2:
                w_half_crop = 30
                h_half_crop = 20
                zoom_fac = 4
            elif self.zoomcycle == 3:
                w_half_crop = 15
                h_half_crop = 10
                zoom_fac = 8
            elif self.zoomcycle == 4:
                w_half_crop = 6
                h_half_crop = 4
                zoom_fac = 16

            w_half_crop = int(size[0] / 2 / self.rescaling_factor/zoom_fac)
            h_half_crop = int(size[1] / 2 / self.rescaling_factor/zoom_fac)

            if self.orig_img_left is not None:
                tmp_left = self.orig_img_left.crop((x_orig - w_half_crop, y_orig - h_half_crop,
                                          x_orig + w_half_crop, y_orig + h_half_crop))

                self.zimg_left = ImageTk.PhotoImage(tmp_left.resize(size))
                self.zimg_id_left = self.canvas_left.create_image(event.x, event.y, image=self.zimg_left)

            if self.orig_img_right is not None:
                tmp_right = self.orig_img_right.crop((x_orig - w_half_crop, y_orig - h_half_crop,
                                          x_orig + w_half_crop, y_orig + h_half_crop))
    
                
    
                self.zimg_right = ImageTk.PhotoImage(tmp_right.resize(size))
    
                self.zimg_id_right = self.canvas_right.create_image(event.x, event.y, image=self.zimg_right)

    def start_widgets(self):
        self.topbot()

    def topbot(self):
        self.canvas_left.pack(fill='both', expand=1, side='top')
        self.canvas_right.pack(fill='both', expand=1, side='bottom')

    def leftright(self):
        self.canvas_left.pack(fill='both', expand=1, side='left')
        self.canvas_right.pack(fill='both', expand=1, side='right')

    def single(self):
        self.canvas_left.pack(fill='both', expand=1, side='left')
        self.canvas_right.pack_forget()
        
    def set_image(self, im, i):
        if im is None: return

        if i == 0:
            canvas = self.canvas_left
        elif i == 1:
            canvas = self.canvas_right
        else:
            print("i can only be 0 or 1")
            return -1
        
        if isinstance(im, Image.Image): #type(im) == type(Image):
            img = im
        else:
            if np.max(im) <= 1:
                im = (im * 255).astype(np.uint8)
            elif np.max(im) <= 255:
                im = im.astype(np.uint8)
    
            self.shape_orig = np.shape(im) # to keep height, width order
            
            if len(self.shape_orig) == 2:    # greyscale
                img = Image.fromarray(im)
            elif np.shape(im)[-1] == 1: # greyscale
                img = Image.fromarray(im)
            elif np.shape(im)[-1] == 3:
                img = Image.fromarray(im, 'RGB')
            else:
                raise AssertionError('depth should be 1 or 3')
        
        if 1:
            self.shape_canvas = (canvas.winfo_height(), canvas.winfo_width())
            
            if i == 0:
                self.orig_img_left = img
            if i == 1:
                self.orig_img_right = img
    
            ratio_canvas = self.shape_canvas[0] / self.shape_canvas[1]  # Height vs width
            ratio_image = self.shape_orig[0] / self.shape_orig[1]
    
            if ratio_image >= ratio_canvas:
                # image will be put on LEFT of canvas
                rescaling_factor = self.shape_canvas[0] / self.shape_orig[0]
                new_shape = (self.shape_canvas[0], int(np.ceil(self.shape_canvas[0] / ratio_image)))
            else:
                # image will be on top of canvas
                rescaling_factor = self.shape_canvas[1]/self.shape_orig[1]
                new_shape = (int(np.ceil(self.shape_canvas[1] * ratio_image)), self.shape_canvas[1])
                
            self.rescaling_factor = rescaling_factor
    
            self.new_shape = new_shape
            img = img.resize((new_shape[1], new_shape[0]), Image.ANTIALIAS)
        
        canvas.img = ImageTk.PhotoImage(img)
        
        canvas.create_image(0, 0, image=canvas.img, anchor="nw")

        # canvas.update_idletasks()


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
