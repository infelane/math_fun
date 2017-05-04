# Abstract class
from abc import ABCMeta

class Basic(metaclass=ABCMeta):
    @staticmethod
    # @abstractmethod
    def my_abstract_staticmethod(self, x):
        ...
    
    def set_input(self):
        ...
    
    def get_output(self):
        ...
    
    def new_func(self):
        ...