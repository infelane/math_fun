import tkinter as tk
# import tkinter.ttk as ttk
from tkinter.ttk import Frame, Style, Notebook
from tkinter import Button, Label, Entry, Tk, BOTH, Frame as TkFrame
import numpy as np

from f2017_08.GUI.main_gui import PlotPanel, PanelImages
from f2017_09.gui.data_structs import GUIData, ImageStruct
from link_to_soliton.paint_tools import image_tools


class ButtonImage(Button):
    def __init__(self, image_struct: ImageStruct, master = None, **args):
        super(self.__class__, self).__init__(master = master, **args)
        self.image_struct = image_struct
        
    def get_image_struct(self):
        return self.image_struct


class FrameSetttingsSub(TkFrame):
    def __init__(self, master = None, **kwargs):
        bg_frame = 'lightgrey'
        
        # args_fixed = **(

        # args_fixed
        
        super(self.__class__, self).__init__(master = master, **kwargs,
                                             bd=2, relief="sunken", bg=bg_frame)


class FrameSettings(Frame):
    def __init__(self, master = None, **args):
        super(self.__class__, self).__init__(master = master, **args)
        # self.pack()
       
        bg_frame = 'lightgrey'

        """ inputs """
        self.frame_inputs = TkFrame(self, bd = 2, relief = "sunken", bg = bg_frame)
        self.frame_inputs.grid(row = 0)

        Label(self.frame_inputs, text='Inputs', bg = bg_frame).pack(side='top')

        """ annotations """
        self.frame_annot = FrameSetttingsSub(self)
        self.frame_annot.grid(row=1)

        Label(self.frame_annot, text='Annotations', bg=bg_frame).pack(side='top')

        """ train """
        # , bg = 'yellow'
        self.frame_training = FrameSetttingsSub(self)
        self.frame_training.grid(row=2)

        Label(self.frame_training, text='Learning', bg=bg_frame).pack(side='top')
        Button(self.frame_training, text='Start training').pack(side="left", padx=5)

        """ output """
        self.frame_output = FrameSetttingsSub(self)
        self.frame_output.grid(row=3)

        Label(self.frame_output, text='Results', bg=bg_frame).pack(side='top')
        Button(self.frame_output, text='Show').pack(side="left", padx=5)

    def set_input_buttons(self, images):
        
        for image_i in images:
            button_i = ButtonImage(image_i, self.frame_inputs, text=image_i.title)
            button_i.pack(side="left", padx=5)
            
            lamda_button = lambda event, button_ii = button_i: self.button_func(button_ii.get_image_struct())
            
            button_i.bind('<Button-1>', lamda_button)

    def set_annot_buttons(self, images):
        for image_i in images:
            button_i = ButtonImage(image_i, self.frame_annot, text=image_i.title)
            button_i.pack(side="left", padx=5)
        
            lamda_button = lambda event, button_ii=button_i: self.button_func(button_ii.get_image_struct())
        
            button_i.bind('<Button-1>', lamda_button)
            
    def set_func(self, func):
        self.button_func = func
        
        
class PanelImages2(Frame):
    def __init__(self, master = None, **args):
        super(self.__class__, self).__init__(master = master, **args)

        path = '/home/lameeus/Pictures/lameeus_pasfoto.png'
        from PIL import ImageTk, Image
        img = ImageTk.PhotoImage(Image.open(path))
        panel = tk.Label(self, image=img)
        panel.pack(side="bottom", fill="both", expand="yes")

        def openfn():
            from tkinter import filedialog
            # filename = filedialog.askopenfilename(title='open')
            # return filename

            folder = '/home/lameeus/data/ghent_altar/input/'
            image_clean2 = ImageStruct('clean2', folder + '19_clean.tif')
            
            return folder + '19_clean.tif'
            # return '/home/lameeus/Pictures/lameeus_pasfoto.png'

        def open_img():
            x = openfn()
            # img = Image.open(x)
            img = image_tools.path2im(x)
            img = img*255
            img = img.astype(np.uint8)
            img = Image.fromarray(img, 'RGB')
            
            # img = img.resize((250, 250), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)
            panel = Label(self, image=img, height = 500, width = 500)
            panel.image = img
            panel.pack(side = "bottom", fill = "both", expand = "yes")

        Button(self, text='open image', command=open_img).pack()
        
    def set_image(self, im):
        # from PIL import ImageTk
        ...
        # img = ImageTk.PhotoImage(im)
        #
        # panel = tk.Label(root, image=img)
        # panel.
        # panel.pack(side="bottom", fill="both", expand="yes"
        
        
def resize_im(im):
    from PIL import Image

    im_new = Image.fromarray(im, 'RGB')
    return im_new.resize((250, 250), Image.ANTIALIAS)
    

class FramePlot(Frame):
    def __init__(self, master = None, **args):
        super(self.__class__, self).__init__(master = master, **args)

        self.panel_images = PanelImages(self)
        
        # self.panel_images2 = PanelImages2(self)
        # self.panel_images2.pack()

    def set_image(self, im):
        # im = image_tools.path2im(path)

        # img = Image.fromarray(img, 'RGB')

        # img = img.resize((250, 250), Image.ANTIALIAS)

        # im = resize_im(im)
        
        self.panel_images.clear_image()
        self.panel_images.set_image(im)
        self.panel_images.update()

        # self.panel_images2.set_image(im)


class MainWindow(Frame):
    def __init__(self, master):
        gui_data = GUIData()
        
        s = Style()
        s.configure('.', font=('Helvetica', 12), background='grey')
        s.configure('My1.TFrame', background='cyan')
        s.configure('My2.TFrame', background='maroon')
        s.configure('ugent.TFrame', background = 'white')
        
        # Frame.__init__(self, master)
        super(MainWindow, self).__init__(master=master, style='My1.TFrame')

        n = Notebook(master)
        n.pack(fill=tk.BOTH, expand=1)
        
        f1 = Frame(n, style='ugent.TFrame')  # first page, which would get widgets gridded into it
        n.add(f1, text='Iedereen UGent')
        
        frame_left = FrameSettings(f1, style='My1.TFrame')
        frame_left.grid(row=0, column=0, sticky="NESW")
        frame_right = FramePlot(f1, style='My2.TFrame')
        frame_right.grid(row=0, column=1
                         , sticky = "NESW"
                         )

        frame_left.set_input_buttons(gui_data.get_input_images())
        frame_left.set_annot_buttons(gui_data.get_annot_images())
        
        frame_right.set_image(gui_data.image_annot.get_im())
        
        def func_set_im(image_struct):
            print(image_struct.title)
            frame_right.set_image(image_struct.get_im())
        # func_set_im = lambda image_struct :

        frame_left.set_func(func_set_im)

        # label = Label(frame_left, text="First")
        # label.pack()
        
        # Label(frame_right, text="Second").pack()
        
    
     


        # checkbutton.grid(columnspan=2, sticky=W)

        # image.grid(row=0, column=2, columnspan=2, rowspan=2,
        #            sticky=W + E + N + S, padx=5, pady=5)

        # button1.grid(row=2, column=2)
        # button2.grid(row=2, column=3)
        
        
        # f1.pack(fill = tk.BOTH, expand = 1)
        
        
        # self.parent = master
        # self.pack(fill = tk.BOTH)
        # ...
        # self.make_widgets()
        
        # s = Style()
        # s.configure('My.TFrame', background='red')
        #

        #
        # # frame_left = FrameSettings(master = self)
        # # frame_right = FramePlot(master = self)
        #
        # frame_left.grid(row=0, column=1)
        # frame_right.grid(row=1, column=1)

        # label = Label(master, text="First")
        # label.pack()

        # Label(master, text="First").grid(row=0, sticky = 'WENS')
        # Label(master, text="Second").grid(row=1, sticky = 'WENS')
        #
        # e1 = Entry(master)
        # e2 = Entry(master)
        #
        # e1.grid(row=0, column=1, sticky = 'WENS')
        # e2.grid(row=1, column=1, sticky = 'WENS')
        
        # frame_left.pack()
        

    def make_widgets(self):
        # don't assume that self.parent is a root window.
        # instead, call `winfo_toplevel to get the root window
        # self.winfo_toplevel().title("Simple Prog")
    
        # this adds something to the frame, otherwise the default
        # size of the window will be very small
        # label = Entry(self)
        # label.pack(side="top", fill="x")
        ...
    

def main():
    window = tk.Tk()
    
    def center_window(window, width=300, height=200):
        # get screen width and height
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
    
        if 1:
            # calculate position x and y coordinates
            x = (screen_width / 2) - (width / 2)
            y = (screen_height / 2) - (height / 2)
        elif 0:
            x = 0
            y = 0

        window.geometry('%dx%d+%d+%d' % (width, height, x, y))

    if 0:
        # Big screen
        center_window(window, width=1920, height=1080)
    else:
        center_window(window)
        
    # window.title = 'Iedereen Ugent'
    start = MainWindow(window)
    window.mainloop()
    

if __name__ == '__main__':
    main()
