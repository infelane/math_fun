from tkinter.ttk import Button

class ButtonPlot(Button):
    def __init__(self, master = None, **args):
        super(self.__class__, self).__init__(master = master, **args)
        
    def set_plot_func(self, func):
        """ func to show the image"""
        
        lambda0 = lambda: func(self.func_set_im(), self.title, 0)
        self.bind('<Button-1>', lambda event: lambda0())
        lambda1 = lambda: func(self.func_set_im(), self.title, 1)
        self.bind('<Button-3>', lambda event: lambda1())
        
    def set_im(self, func, title):
        """ func to get image"""
        
        self.func_set_im = func
        self.title = title
