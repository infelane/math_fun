import tkinter as tk
# import tkinter.ttk as ttk
from tkinter.ttk import Frame, Style, Notebook, Button
from tkinter import Label, Entry, Tk, Frame as TkFrame, StringVar, Canvas, OptionMenu
import numpy as np
from PIL import ImageTk, Image

from f2017_08.GUI.main_gui import PlotPanel, PanelImages
from f2017_09.gui.data_structs import GUIData, ImageStruct
from link_to_soliton.paint_tools import image_tools
from f2017_09.gui import frames, network_stuff

bg_frame = 'lightgrey'


class ButtonImage(Button):
    def __init__(self, image_struct: ImageStruct, master = None, **args):
        super(self.__class__, self).__init__(master = master, **args)
        self.image_struct = image_struct
        
    def get_image_struct(self):
        return self.image_struct


class FrameSetttingsSub(TkFrame):
    def __init__(self, master = None, **kwargs):
        
        
        super(self.__class__, self).__init__(master = master, **kwargs,
                                             relief="flat",
                                             # bd=2, relief="sunken",
                                             bg=bg_frame)
        
        self.button_frame = TkFrame(self, bg = bg_frame)
        self.button_frame.pack(side = 'bottom')
        
    def get_button_frame(self):
        return self.button_frame


class FrameSettings(Frame):
    def __init__(self, master = None, **args):
        super(self.__class__, self).__init__(master = master, **args)
        
        self.init_frame_input()
        self.init_frame_labeling()
        self.init_frame_loading()
        self.init_frame_training()
        self.init_frame_output()
        
        self.init_frame_print()
        
        self.grid_rowconfigure((0, 1, 2, 3, 4), weight = 0, minsize = 0, pad = 10, uniform = False)
        self.grid_rowconfigure((5), weight = 10, minsize = 0, pad = 10, uniform = True)
        self.grid_columnconfigure(0, weight = 0, minsize = 0, pad = 10, uniform = True)

    def __set_x_buttons(self, images, frame_part):
        
        def lambda_button1(button_ii):
            self.set_text('Loading')
            self.button_func(button_ii.get_image_struct())
            self.set_text('Done')
        
        for image_i in images:
            button_i = ButtonImage(image_i, frame_part.get_button_frame(), text=image_i.title)
            button_i.pack(side="left", padx=5, pady=5)
        
            lambda_button = lambda event, button_ii=button_i: lambda_button1(button_ii)
        
            button_i.bind('<Button-1>', lambda_button)
            
    def __set_sub_frame(self, text, row = None):
        sub_frame = FrameSetttingsSub(self)
        sub_frame.grid(row=row, sticky="new", padx = 10, pady = 5)
        Label(sub_frame, text = text, bg=bg_frame).pack(side='top', padx =5, pady = 5)
        
        return sub_frame
        
    def __create_button(self, master, text = None):
        b = Button(master, text=text)
        b.pack(side="bottom", padx=5, pady=5)
        return b
    
    def init_frame_input(self):
        """ inputs """
        
        text = 'Inputs'
        self.frame_inputs = self.__set_sub_frame(text, 0)
        
        input_sets_variable = StringVar(self.frame_inputs)
        set_frame = TkFrame(self.frame_inputs, background = bg_frame)
        set_frame.pack(ipadx=10)
        
        l = Label(set_frame, text = 'set:', bg=bg_frame)
        l.pack(anchor = 'center', side = 'left')
        om = OptionMenu(set_frame, input_sets_variable, None)
        om.pack(anchor = 'center', side = 'right')
        
    def init_frame_labeling(self):
        """ Annotations """
        text = 'Annotations'
        self.frame_annot = self.__set_sub_frame(text, 1)
        
    def init_frame_loading(self):
        """ Loading """
        text = 'Loading'
        self.frame_loading = self.__set_sub_frame(text, 2)

        # options = [
        #     "None",
        #     "pretrained",
        # ]  # etc
        #
        # def OptionMenu_SelectionEvent(event):  # I'm not sure on the arguments here, it works though
        #     ## do something
        #
        #     self.loading_option_func(self.loading_variable.get())
        #
        #     print(self.loading_variable.get())

        self.loading_variable = StringVar(self.frame_loading)
        # self.loading_variable.set(options[1])  # default value
       
        # self.loading_option_menu = OptionMenu(self.frame_loading, self.loading_variable, *options, command=OptionMenu_SelectionEvent)
        self.loading_option_menu = OptionMenu(self.frame_loading, self.loading_variable, None)
        self.loading_option_menu.pack()
        
    def init_frame_training(self):
        """ train """
        
        text = 'Learning'
        self.frame_training = self.__set_sub_frame(text, 3)
        self.button_training = self.__create_button(self.frame_training, 'Start training')
        
    def init_frame_output(self):
        """ Annotations """
        text = 'Results'
        self.frame_output = self.__set_sub_frame(text, 4)
        b = self.__create_button(self.frame_output, text='Inference')

        b.bind('<Button-1>', lambda event : self.button_inference_func())

        self.result_sets_variable = StringVar(self.frame_output)
        self.results_sets_option_menu = OptionMenu(self.frame_output, self.result_sets_variable, None)
        self.results_sets_option_menu.pack()

    def init_frame_print(self):
        """ For printing notes """
        
        sub_frame = FrameSetttingsSub(self)
        sub_frame.grid(row=5, sticky = 'sew', pady = 5, padx = 10)

        self.textvariable = StringVar()

        self.textvariable.set('Welcome')
        
        text_box = Entry(sub_frame, textvariable = self.textvariable)
        text_box.pack(side='bottom', padx = 5, pady = 5, fill = 'both')
        
        self.text_box_info = text_box

    def set_input_buttons(self, images):
        self.__set_x_buttons(images, self.frame_inputs)

    def set_annot_buttons(self, images):
        self.__set_x_buttons(images, self.frame_annot)
        
    def set_output_buttons(self, images):
        self.__set_x_buttons(images, self.frame_output)
        
    def set_func(self, func):
        self.button_func = func
        
    def set_func_inference(self, func):
        self.button_inference_func = func
        
    def set_text(self, text):
        self.textvariable.set(text)
        self.text_box_info.update_idletasks()
        
    def set_loading_options(self, names, func):
        
        # Reset var and delete all old options
        menu = self.loading_option_menu['menu']
        menu.delete(0, 'end')

        for name_i in names:
            menu.add_command(label=name_i,
                             command=lambda value=name_i: self.loading_variable.set(value))

        def callback(sv):
            print(sv.get())
            func(sv.get())

        self.loading_variable.trace("w", lambda name, index, mode, sv=self.loading_variable: callback(sv))
        self.loading_variable.set(names[0])
        
    def set_training_func(self, func):
    
        lambda_button = lambda event : func()
        self.button_training.bind('<Button-1>', lambda_button)
        
    def set_result_sets(self, names, func):
        # Reset var and delete all old options
        menu = self.results_sets_option_menu['menu']
        menu.delete(0, 'end')

        for name_i in names:
            # TODO
            ...
            menu.add_command(label=name_i,
                             command=lambda value=name_i: self.result_sets_variable.set(value))

        def callback(sv):
            print(sv.get())
            func(sv.get())

        self.result_sets_variable.trace("w", lambda name, index, mode, sv=self.result_sets_variable: callback(sv))
        self.result_sets_variable.set(names[0])
        
        
class PanelImages2(Frame):
    def __init__(self, master = None, **args):
        super(self.__class__, self).__init__(master = master, **args)

        path = '/home/lameeus/Pictures/lameeus_pasfoto.png'
        img = ImageTk.PhotoImage(Image.open(path))
        panel = Label(self, image=img)
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

        self.panel_images = frames.PanelImages3(self)
        
        # self.panel_images2 = PanelImages2(self)
        # self.panel_images2.pack()

    def set_image(self, im):
        self.panel_images.set_image(im)


class MainWindow(Frame):
    def __init__(self, master):
        gui_data = GUIData()
        
        s = Style()
        
        print(s.theme_names())
        
        s.theme_use('clam')
        # s.theme_use('alt')
        # s.theme_use('classic')
        
        # font = ('fixedsys', 12)
        # font = ('system 10', 12)
        # s.configure('.TFrame', font=font, background='grey')
        # s.configure('.TLabel', font=font, background='white')
        # s.configure('.', font=font, background='white')
        s.configure('My1.TFrame', background='cyan')
        s.configure('My2.TFrame', background='maroon')
        s.configure('ugent.clam') #, background = 'white')
        
        # Frame.__init__(self, master)
        super(MainWindow, self).__init__(master=master, style='My1.TFrame')

        n = Notebook(master)
        n.pack(fill='both', expand=1)
        
        f1 = Frame(n, style='ugent.TFrame')  # first page, which would get widgets gridded into it
        n.add(f1, text='Iedereen UGent')
        
        frame_left = FrameSettings(f1) #, style='ugent.TFrame')
        frame_right = FramePlot(f1, style='My2.TFrame')

        if 1:
            frame_left.grid(row=0, column=0, sticky="NESW")
            frame_right.grid(row=0, column=1,
                             sticky = "NESW",
                             )

            f1.grid_columnconfigure(1, weight = 1)
            f1.grid_rowconfigure(0, weight=1)
        else:
            frame_left.pack(anchor='n', fill='both', expand=False, side='left' )
            frame_right.pack(anchor='n', fill='both', expand=True, side='left' )


        frame_left.set_input_buttons(gui_data.get_input_images())
        frame_left.set_annot_buttons(gui_data.get_annot_images())
        frame_left.set_output_buttons(gui_data.get_output_images())
        
        frame_right.set_image(gui_data.get_input_images()[0].get_im())
        
        def func_set_im(image_struct):
            print(image_struct.title)
            frame_right.set_image(image_struct.get_im())
        # func_set_im = lambda image_struct :

        frame_left.set_func(func_set_im)
        
        network = network_stuff.NetworkStuff()
        
        def func_set_inference():
            print('test test test')
            
            im = network.inference()
            
            frame_right.set_image(im)
            
            # TODO something with after!
            # self.after(10, None)
        
        frame_left.set_func_inference(func_set_inference)

        frame_left.set_loading_options(network.loading_options(),
                                       lambda name : network.load_network(name))

        epoch_func = func_set_inference

        func_print = frame_left.set_text
        
        frame_left.set_training_func(lambda : network.training(epoch_func, func_print = func_print))

        
        #TODO
        frame_left.set_result_sets(gui_data.get_names_sets(),
                                   lambda name: network.set_name_set(name)
                                   )

        self.make_widgets()
    
        if 0:
            class TextOut(tk.Text):
    
                def write(self, s):
                    self.insert(tk.CURRENT, s)
    
                def flush(self):
                    pass
    
            import sys
            text = TextOut(frame_left)
            sys.stdout = text
            text.grid(row = 6)

    def make_widgets(self):
        # don't assume that self.parent is a root window.
        # instead, call `winfo_toplevel to get the root window

        self.winfo_toplevel().title("Iedereen UGent DEMO")

 
        
def main():
    window = Tk()
    
    def center_window(window, width=300, height=200):
        # get screen width and height
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
    
        if 0:
            # calculate position x and y coordinates
            x = (screen_width / 2) - (width / 2)
            y = (screen_height / 2) - (height / 2)
        elif 1:
            x = 1920//2-1
            y = 0

        window.geometry('%dx%d+%d+%d' % (width, height, x, y))

    if 1:
        # Big screen
        center_window(window, width=1920//2, height=1200)
    else:
        center_window(window)
        
    # window.title = 'Iedereen Ugent'
    start = MainWindow(window)

    # start.master.title("Simple Prog")

    window.mainloop()
    # window.destroy()
    

if __name__ == '__main__':
    main()
