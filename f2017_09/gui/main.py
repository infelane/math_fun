import tkinter as tk
# import tkinter.ttk as ttk
from tkinter.ttk import Frame, Style, Notebook, Button
from tkinter import Label, Entry, Tk, Frame as TkFrame, StringVar, Canvas, OptionMenu
import numpy as np
from PIL import ImageTk, Image

from f2017_09.gui.data_structs import GUIData, ImageStruct, InpaintingData
from link_to_soliton.paint_tools import image_tools
from f2017_09.gui import frames, network_stuff, general_widgets

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
    
    def delete_buttons(self):
        for child in self.get_button_frame().winfo_children():
            child.destroy()


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
    
        frame_part.delete_buttons()
    
        for child in frame_part.get_button_frame().winfo_children():
            child.destroy()
        
        def lambda_button1(button_ii, i):
            self.set_text('Loading')
            self.button_func(button_ii.get_image_struct(), i)
            self.set_text('Done')
            
        for image_i in images:
            button_i = ButtonImage(image_i, frame_part.get_button_frame(), text=image_i.title)
            button_i.pack(side="left", padx=5, pady=5)
            
            # TODO cleaner way
            # b = self.__create_button_plot(self.frame_output, text='Inference')
            # l2 = lambda: self.func_output_im()
            # b.set_im(l2)
        
            lambda_button = lambda event, button_ii=button_i, i = 0: lambda_button1(button_ii, i)
            button_i.bind('<Button-1>', lambda_button)
            lambda_button = lambda event, button_ii=button_i, i = 1: lambda_button1(button_ii, i)
            button_i.bind('<Button-3>', lambda_button)
            
    def __set_sub_frame(self, text, row = None):
        sub_frame = FrameSetttingsSub(self)
        sub_frame.grid(row=row, sticky="new", padx = 10, pady = 5)
        Label(sub_frame, text = text, bg=bg_frame).pack(side='top', padx =5, pady = 5)
        
        return sub_frame
        
    def __create_button(self, master, text = None):
        b = Button(master, text=text)
        b.pack(side="bottom", padx=5, pady=5)
        return b
    
    def __create_button_plot(self, master, text = None):
        b = general_widgets.ButtonPlot(master, text = text)
        b.pack(side="bottom", padx=5, pady=5)
        # This is always the same
        l = lambda im, i: self.func_imshow(im, i)
        b.set_plot_func(l)
        return b
    
    def init_frame_input(self):
        """ inputs """
        
        text = 'Inputs'
        self.frame_inputs = self.__set_sub_frame(text, 0)

        self.input_sets_variable = StringVar(self.frame_inputs)
        
        set_frame = TkFrame(self.frame_inputs, background = bg_frame)
        set_frame.pack(ipadx=10)
    
        l = Label(set_frame, text = 'set:', bg=bg_frame)
        l.pack(anchor = 'center', side = 'left')
        self.input_sets_option_menu = OptionMenu(set_frame, self.input_sets_variable, None)
        self.input_sets_option_menu.pack(anchor = 'center', side = 'right')
        
    def init_frame_labeling(self):
        """ Annotations """
        text = 'Annotations'
        self.frame_annot = self.__set_sub_frame(text, 1)
        
    def init_frame_loading(self):
        """ Loading """
        text = 'Load the neural network'
        self.frame_loading = self.__set_sub_frame(text, 2)

        set_frame = TkFrame(self.frame_loading, background=bg_frame)
        set_frame.pack(ipadx=10)

        l = Label(set_frame, text='weights:', bg=bg_frame)
        l.pack(anchor='center', side='left')
        
        self.loading_variable = StringVar(self.frame_loading)

        self.loading_option_menu = OptionMenu(set_frame, self.loading_variable, None)
        self.loading_option_menu.pack(anchor='center', side='right')

        # l = Label(set_frame, text='set:', bg=bg_frame)
        # l.pack(anchor='center', side='left')
        # self.input_sets_option_menu = OptionMenu(set_frame, self.input_sets_variable, None)
        # self.input_sets_option_menu.pack(anchor='center', side='right')
        
    def init_frame_training(self):
        """ train """
        
        text = 'Learning'
        self.frame_training = self.__set_sub_frame(text, 3)
        # training function is different, so no fancy button_plot
        self.button_training = self.__create_button(self.frame_training, 'Start training')
        
    def init_frame_output(self):
        """ Annotations """
        text = 'Results'
        self.frame_output = self.__set_sub_frame(text, 4)
        
        b = self.__create_button_plot(self.frame_output, text = 'Inference')
        l2 = lambda: self.func_output_im()
        b.set_im(l2)
        
        self.result_sets_variable = StringVar(self.frame_output)
        # self.results_sets_option_menu = OptionMenu(self.frame_output, self.result_sets_variable, None)
        # self.results_sets_option_menu.pack(
        #
        # )
        
        set_frame = TkFrame(self.frame_output, background = bg_frame)
        set_frame.pack(ipadx=10)
        
        l = Label(set_frame, text = 'set:', bg=bg_frame)
        l.pack(anchor = 'center', side = 'left')
        self.results_sets_option_menu = OptionMenu(set_frame, self.result_sets_variable, None)
        self.results_sets_option_menu.pack(anchor = 'center', side = 'right')

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
        
    def set_func_imshow(self, func):
        self.func_imshow = func
        
    def set_func_output_im(self, func):
        self.func_output_im = func
        
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
        """ the function is different so no fancy shortcuts with image showing """
        lambda_button0 = lambda event : func(i =0)
        self.button_training.bind('<Button-1>', lambda_button0)
        lambda_button1 = lambda event: func(i = 1)
        self.button_training.bind('<Button-3>', lambda_button1)
        
    def set_input_sets(self, sets):
        # Reset var and delete all old options
        menu = self.input_sets_option_menu['menu']
        menu.delete(0, 'end')

        sets_dict = {set_i.name: set_i for set_i in sets}
        
        for set_i in sets:
            menu.add_command(label=set_i.name,
                             command=lambda name_i=set_i.name: self.input_sets_variable.set(name_i)
                             )
            
        def callback(sv):
            print('inputs sets variable altered')

            set_i = sets_dict[sv.get()]

            self.set_input_buttons(set_i.get_input_images())
            self.set_annot_buttons(set_i.get_annot_images())
            self.set_output_buttons(set_i.get_output_images())
        
        self.input_sets_variable.trace("w", lambda name, index, mode, sv=self.input_sets_variable: callback(sv))
        self.input_sets_variable.set(sets[0].name)
        
    def set_result_sets(self, names, func):
        # Reset var and delete all old options
        menu = self.results_sets_option_menu['menu']
        menu.delete(0, 'end')

        for name_i in names:
            menu.add_command(label=name_i,
                             command=lambda value=name_i: self.result_sets_variable.set(value))

        def callback(sv):
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


class FramePlotDual(Frame):
    def __init__(self, master=None, **args):
        super(self.__class__, self).__init__(master=master, **args)
        
        self.panel_images = frames.PanelDualImage(self)
    
    def set_image(self, im, i):
        self.panel_images.set_image(im, i)
        
        
class ButtonsAll(Frame):
    def __init__(self, master, **args):
        super(self.__class__, self).__init__(master=master, **args)
        
        self.buttons = []

        b = general_widgets.ButtonPlot(self, text = 'Original')
        b.set_im(lambda *args: self.inpainting_set.get_orig(*args))
        self.button_stuff(b)
        
        b = general_widgets.ButtonPlot(self, text = 'Mask')
        b.set_im(lambda *args: self.inpainting_set.get_map(*args))
        self.button_stuff(b)

        b = general_widgets.ButtonPlot(self, text='Result')
        b.set_im(lambda *args: self.inpainting_set.get_result(*args))
        self.button_stuff(b)

        # only show when given
        b = general_widgets.ButtonPlot(self, text='Real restoration')
        b.set_im(lambda *args: self.inpainting_set.get_restored(*args))
        self.button_stuff(b)
        
    def set_last_button_state(self, bool):
        if bool:
            self.buttons[-1].pack()
        else:
            self.buttons[-1].pack_forget()

    def button_stuff(self, b):
        b.pack()
        b.set_plot_func(lambda *args: self.func(*args))
        self.buttons.append(b)
        
    def set_inpainting_set(self, inpainting_set):
        self.inpainting_set = inpainting_set
        
    def set_func(self, func):
        self.func = func
        
    def set_button(self, i, func):
        self.buttons[i].bind('<Button-1>', lambda event: func())
        
    def enable_real(self, bool = False):
        if bool:
            self.buttons[3].config(state = 'normal')
        else:
            self.buttons[3].config(state = 'disabled')

        
class FrameInfoDual(Frame):
    def __init__(self, master=None, **args):
        super(self.__class__, self).__init__(master=master, **args)
        
        frame_set = TkFrame(self, background=bg_frame)
        
        l = Label(frame_set, text='set:', bg=bg_frame)
        self.sets_var = StringVar(frame_set)
        self.om = OptionMenu(frame_set, self.sets_var, None)
        
        self.buttons1 = ButtonsAll(self)
        # self.buttons2 = ButtonsAll(self)
        
        def pack1(a):
            a.pack(padx=10, pady=10, ipadx=10, ipady=10, anchor='center', side='top')
        
        # frame_set.pack(ipadx=10)
        pack1(frame_set)
        pack1(self.buttons1)
        
        l.pack(anchor='e', side = 'left')
        self.om.pack(anchor='w', side='right')
        
        self.start_widgets()

    def start_widgets(self):
        ...
        
    def set_options(self, options, func_show_im):
        # # Reset var and delete all old options
        menu = self.om['menu']
        menu.delete(0, 'end')

        self.buttons1.set_func(func_show_im)

        options_dict = {opt_i.name : opt_i for opt_i in options}

        # def func(opt_i):
        #     self.sets_var.set(opt_i.name)
    
        for opt_i in options:
            name_i = opt_i.name
            menu.add_command(label=name_i,
                             command=lambda name_ii=name_i: self.sets_var.set(name_ii))
            
        def callback(sv):
            print(sv.get())
            name_i = sv.get()
            opt_i = options_dict[name_i]
            self.buttons1.set_inpainting_set(opt_i)
    
            if opt_i.get_restored() is None:
                self.buttons1.set_last_button_state(False)
            else:
                self.buttons1.set_last_button_state(True)
            
        # set to first value
        self.sets_var.trace("w", lambda name, index, mode, value= self.sets_var: callback(value))
        self.sets_var.set(options[0].name)

        # self.sets_var.trace("w", lambda name, index, mode, value=self.sets_var: callback(value))
        # self.sets_var.trace("w", lambda name, index, mode:None)
        
        
        # # Reset var and delete all old options
        # menu = self.results_sets_option_menu['menu']
        # menu.delete(0, 'end')
        #
        # for name_i in names:
        #     # TODO
        #     menu.add_command(label=name_i,
        #                      command=lambda value=name_i: self.result_sets_variable.set(value))
        #
        # def callback(sv):
        #     print(sv.get())
        #     func(sv.get())
        #
        # self.result_sets_variable.trace("w", lambda name, index, mode, sv=self.result_sets_variable: callback(sv))
        # self.result_sets_variable.set(names[0])
        
        
class FrameInpainting(Frame):
    def __init__(self, master = None, **args):
        super(self.__class__, self).__init__(master = master, **args)

        self.frame_left = FrameInfoDual(self)
        self.frame_right = FramePlotDual(self)

        self.frame_right.pack(anchor='nw', side='right', fill='both', expand=True)
        self.frame_left.pack(anchor='ne', side='left')
        
        self.start_widgets()
        
    def start_widgets(self):
        inpainting_data = InpaintingData()
        self.frame_left.set_options(inpainting_data.get_sets(), self.frame_right.set_image)
        
        
class FrameHowTo(Frame):
    def __init__(self, master = None, **args):
        super(self.__class__, self).__init__(master = master, **args)
        
        def pack1(a):
            a.pack(fill = 'both', expand = 0, pady = 10, padx = 10)
            
        text = 'Left click: show image'
        pack1(Label(self, text = text))
        text = 'Right click: show image at second canvas'
        pack1(Label(self, text=text))
        text = 'Scroll: zoom in/out'
        pack1(Label(self, text=text))

        
class FrameLosses(Frame):
    def __init__(self, master = None, **args):
        super(self.__class__, self).__init__(master = master, **args)
        
        gui_data = GUIData()

        frame_left = FrameSettings(self)  # , style='ugent.TFrame')
        
        if 0:
            frame_right = FramePlot(self, style='My2.TFrame')
        else:
            frame_right = FramePlotDual(self, style='My2.TFrame')
        if 1:
            frame_left.grid(row=0, column=0, sticky="NESW")
            frame_right.grid(row=0, column=1,
                             sticky="NESW",
                             )

            self.grid_columnconfigure(1, weight=1)
            self.grid_rowconfigure(0, weight=1)
        else:
            frame_left.pack(anchor='n', fill='both', expand=False, side='left')
            frame_right.pack(anchor='n', fill='both', expand=True, side='left')

        def func_set_im(image_struct, i = 0):
            print(image_struct.title)
            frame_right.set_image(image_struct.get_im(), i)

        # func_set_im = lambda image_struct :

        frame_left.set_func(func_set_im)

        network = network_stuff.NetworkStuff()

        def func_set_inference(i):
            print('test test test')
    
            im = network.inference()
    
            frame_right.set_image(im, i)
    
            # TODO something with after!
            # self.after(10, None)

        frame_left.set_func_inference(func_set_inference)
        frame_left.set_func_imshow(frame_right.set_image)
        frame_left.set_func_output_im(network.inference)

        frame_left.set_loading_options(network.loading_options(),
                                       lambda name: network.load_network(name))

        # epoch_func = func_set_inference

        func_print = frame_left.set_text
        
        def training_func(i):
            epoch_func = lambda arg = i: func_set_inference(arg)
            network.training(epoch_func, func_print=func_print)
            

        frame_left.set_training_func(training_func)

        # TODO
        frame_left.set_input_sets(gui_data.get_input_sets())
        
        frame_left.set_result_sets(gui_data.get_names_sets(),
                                   lambda name: network.set_name_set(name)
                                   )


class MainWindow(Frame):
    def __init__(self, master):

        
        s = Style()
        
        print(s.theme_names())
        
        s.theme_use('clam')

        font = ('Helvetica', 14)
        font_tab = ('Helvetica', 20)

        s.configure(".", font=font)
        s.configure("TNotebook.Tab", font = font_tab)

        mygreen = "#d2ffd2"
        myred = "#dd0202"
        
        s.configure('My1.TFrame', background='cyan')
        s.configure('My2.TFrame', background='maroon')
        s.configure('ugent.clam') #, background = 'white')
        
        super(MainWindow, self).__init__(master=master, style='My1.TFrame')

        n = Notebook(master)
        n.pack(fill='both', expand=1)
        
        f1 = FrameLosses(n, style='ugent.TFrame')  # first page, which would get widgets gridded into it
        n.add(f1, text='Verf verlies')
        
        f2 = FrameInpainting(n)  # first page, which would get widgets gridded into it
        n.add(f2, text='Inschildering')
        
        f2 = FrameHowTo(n)  # first page, which would get widgets gridded into it
        n.add(f2, text='How to')

        # frame_left = FrameSettings(f1)  # , style='ugent.TFrame')
        # frame_right = FramePlot(f1, style='My2.TFrame')
        #
        # if 1:
        #     frame_left.grid(row=0, column=0, sticky="NESW")
        #     frame_right.grid(row=0, column=1,
        #                      sticky="NESW",
        #                      )
        #
        #     f1.grid_columnconfigure(1, weight=1)
        #     f1.grid_rowconfigure(0, weight=1)
        # else:
        #     frame_left.pack(anchor='n', fill='both', expand=False, side='left')
        #     frame_right.pack(anchor='n', fill='both', expand=True, side='left')
        #
        # frame_left.set_input_buttons(gui_data.get_input_images())
        # frame_left.set_annot_buttons(gui_data.get_annot_images())
        # frame_left.set_output_buttons(gui_data.get_output_images())
        #
        # frame_right.set_image(gui_data.get_input_images()[0].get_im())
        #
        # def func_set_im(image_struct):
        #     print(image_struct.title)
        #     frame_right.set_image(image_struct.get_im())
        #
        # # func_set_im = lambda image_struct :
        #
        # frame_left.set_func(func_set_im)
        #
        # network = network_stuff.NetworkStuff()
        #
        # def func_set_inference():
        #     print('test test test')
        #
        #     im = network.inference()
        #
        #     frame_right.set_image(im)
        #
        #     # TODO something with after!
        #     # self.after(10, None)
        #
        # frame_left.set_func_inference(func_set_inference)
        #
        # frame_left.set_loading_options(network.loading_options(),
        #                                lambda name: network.load_network(name))
        #
        # epoch_func = func_set_inference
        #
        # func_print = frame_left.set_text
        #
        # frame_left.set_training_func(lambda: network.training(epoch_func, func_print=func_print))
        #
        # # TODO
        # frame_left.set_result_sets(gui_data.get_names_sets(),
        #                            lambda name: network.set_name_set(name)
        #                            )
        #
        # self.make_widgets()
        #
        # if 0:
        #     class TextOut(tk.Text):
        #
        #         def write(self, s):
        #             self.insert(tk.CURRENT, s)
        #
        #         def flush(self):
        #             pass
        #
        #     import sys
        #     text = TextOut(frame_left)
        #     sys.stdout = text
        #     text.grid(row = 6)

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
