from tkinter import Frame, Canvas, Label
# from tkinter.ttk import
from PIL import Image, ImageTk
import numpy as np
from tkinter.ttk import Button


class PanelDualImage(object):
    def __init__(self, master):
        
        self.frame_canvases = Frame(master)
        self.frame_canvases.pack(side = 'top', fill = 'both', expand = 1)
        
        self.canvas_left = Canvas(self.frame_canvases)
        self.label_left = Label(self.frame_canvases)
        
        self.canvas_right = Canvas(self.frame_canvases)
        self.label_right = Label(self.frame_canvases)
        
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
        self.rescaling_factor_left = 1
        self.rescaling_factor_right = 1
        self.name_left = 'title 1'
        self.name_right = 'title 2'

        self.canvas_left.bind("<Motion>", self.crop)
        self.canvas_left.bind("<Button-4>", self.zoomer)
        self.canvas_left.bind("<Button-5>", self.zoomer)
        
        self.canvas_right.bind("<Motion>", self.crop)
        self.canvas_right.bind("<Button-4>", self.zoomer)
        self.canvas_right.bind("<Button-5>", self.zoomer)

        self.canvas_left.bind("<Configure>", lambda event: self.set_image(self.orig_img_left, self.name_left, 0) )
        self.canvas_right.bind("<Configure>", lambda event: self.set_image(self.orig_img_right, self.name_right, 1) )

    def zoomer(self, event):
        # windows
        if (event.delta > 0):
            if self.zoomcycle != 4: self.zoomcycle += 1
        elif (event.delta < 0):
            if self.zoomcycle != 0: self.zoomcycle -= 1

        # linux
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

            w_rel_left = x / self.new_shape_left[1]
            h_rel_left = y / self.new_shape_left[0]

            w_rel_right = x / self.new_shape_right[1]
            h_rel_right = y / self.new_shape_right[0]

            w_orig_left = w_rel_left * self.shape_orig_left[1]
            h_orig_left = h_rel_left * self.shape_orig_left[0]

            w_orig_right = w_rel_right * self.shape_orig_right[1]
            h_orig_right = h_rel_right * self.shape_orig_right[0]

            size = 600, 400     # first 300, 200
            
            if self.zoomcycle == 1:
                zoom_fac = 2
            elif self.zoomcycle == 2:
                zoom_fac = 4
            elif self.zoomcycle == 3:
                zoom_fac = 8
            elif self.zoomcycle == 4:
                zoom_fac = 16

            w_half_crop_left = int(size[0] / 2 / self.rescaling_factor_left/zoom_fac)
            h_half_crop_left = int(size[1] / 2 / self.rescaling_factor_left/zoom_fac)

            w_half_crop_right = int(size[0] / 2 / self.rescaling_factor_right / zoom_fac)
            h_half_crop_right = int(size[1] / 2 / self.rescaling_factor_right / zoom_fac)

            if self.orig_img_left is not None:
                tmp_left = self.orig_img_left.crop((w_orig_left - w_half_crop_left, h_orig_left - h_half_crop_left,
                                                    w_orig_left + w_half_crop_left, h_orig_left + h_half_crop_left))

                self.zimg_left = ImageTk.PhotoImage(tmp_left.resize(size))
                self.zimg_id_left = self.canvas_left.create_image(event.x, event.y, image=self.zimg_left)

            if self.orig_img_right is not None:
                tmp_right = self.orig_img_right.crop((w_orig_right - w_half_crop_right, h_orig_right - h_half_crop_right,
                                                      w_orig_right + w_half_crop_right, h_orig_right + h_half_crop_right))
    
                self.zimg_right = ImageTk.PhotoImage(tmp_right.resize(size))
    
                self.zimg_id_right = self.canvas_right.create_image(event.x, event.y, image=self.zimg_right)

    def start_widgets(self):
        self.topbot()

    def topbot(self):
        if 0:
            self.label_left.pack(side='top')
            self.canvas_left.pack(fill='both', expand=1, side='top')
    
            self.canvas_right.pack(fill='both', expand=1, side='bottom')
            self.label_right.pack(side='bottom')
    
        else:
            self.label_left.grid(row=0, column = 0, sticky="nw")
            self.canvas_left.grid(row=1, column = 0, sticky="nesw")
            self.label_right.grid(row=2, column = 0, sticky="nw")
            self.canvas_right.grid(row=3, column = 0, sticky="nesw")

            self.frame_canvases.grid_rowconfigure((0, 2), weight=0)
            self.frame_canvases.grid_rowconfigure((1, 3), weight = 1)
            self.frame_canvases.grid_columnconfigure((0), weight=1)
            self.frame_canvases.grid_columnconfigure((1), weight=0)
        
    def leftright(self):
        if 0:
            self.label_left.pack(side='left')
            self.canvas_left.pack(fill='both', expand=1, side='left')
            self.label_right.pack(side='right')
            self.canvas_right.pack(fill='both', expand=1, side='right')
            
        else:
            self.label_left.grid(row=0, column = 0, sticky="nw")
            self.canvas_left.grid(row=1, column = 0)
            self.label_right.grid(row=0, column = 1, sticky="nw")
            self.canvas_right.grid(row=1, column = 1)
            
            self.frame_canvases.grid_rowconfigure((0, 2, 3), weight=0)
            self.frame_canvases.grid_rowconfigure((1), weight = 1)
            self.frame_canvases.grid_columnconfigure((0, 1), weight=1)

    def single(self):
        if 0:
            self.label_left.pack(side='left')
            self.canvas_left.pack(fill='both', expand=1, side='left')
            self.canvas_right.pack_forget()
            self.label_right.pack_forget()
            
        else:
            self.canvas_right.grid_forget()
            self.label_right.grid_forget()
            
            self.label_left.grid(row=0, column=0, sticky="nw")#, sticky="nesw")
            self.canvas_left.grid(row=1, column=0)#, sticky="nesw")

            self.frame_canvases.grid_rowconfigure((0, 2, 3), weight=0)
            self.frame_canvases.grid_rowconfigure((1), weight = 1)
            self.frame_canvases.grid_columnconfigure((0), weight=1)
            self.frame_canvases.grid_columnconfigure((1), weight=0)
        
    def set_image(self, im, title:str, i):
        
        
        if im is None: return -1

        if i == 0:
            canvas = self.canvas_left
            self.name_left = title
            self.label_left['text'] = title
        elif i == 1:
            canvas = self.canvas_right
            self.name_right = title
            self.label_right['text'] = title
            
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

            shape_orig = np.shape(im)   # to keep height, width order
            if i == 0:
                self.shape_orig_left = shape_orig
            elif i == 1:
                self.shape_orig_right = shape_orig
            
            if len(shape_orig) == 2:    # greyscale
                img = Image.fromarray(im)
            elif np.shape(im)[-1] == 1: # greyscale
                img = Image.fromarray(im)
            elif np.shape(im)[-1] == 3:
                img = Image.fromarray(im, 'RGB')
            else:
                raise AssertionError('depth should be 1 or 3')
        
        self.shape_canvas = (canvas.winfo_height(), canvas.winfo_width())
        
        if i == 0:
            self.orig_img_left = img
        if i == 1:
            self.orig_img_right = img

        ratio_canvas = self.shape_canvas[0] / self.shape_canvas[1]  # Height vs width
        
        if i == 0:
            shape_orig = self.shape_orig_left
        elif i == 1:
            shape_orig = self.shape_orig_right
            
        ratio_image = shape_orig[0] / shape_orig[1]

        if ratio_image >= ratio_canvas:
            # image will be put on LEFT of canvas
            
            rescaling_factor = self.shape_canvas[0] / shape_orig[0]
            new_shape = (self.shape_canvas[0], int(np.ceil(self.shape_canvas[0] / ratio_image)))
        else:
            # image will be on top of canvas
            rescaling_factor = self.shape_canvas[1]/shape_orig[1]
            new_shape = (int(np.ceil(self.shape_canvas[1] * ratio_image)), self.shape_canvas[1])

        if i == 0:
            self.rescaling_factor_left = rescaling_factor
            self.new_shape_left = new_shape
        elif i == 1:
            self.rescaling_factor_right = rescaling_factor
            self.new_shape_right = new_shape
 
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
