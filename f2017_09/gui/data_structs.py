import numpy as np

from link_to_soliton.paint_tools import image_tools


class ImageStruct(object):
    def __init__(self, name:str, path:str):
        self.title = name
        self.path = path
        
    def get_im(self):
        im = image_tools.path2im(self.path)
        im = im*255
        return im.astype(np.uint8)
    

    def __str__(self):
        return self.title + 'at' + self.path
    
    
class SetStruct(object):
    def __init__(self, name: str):
        if name == 'evangelist':
            folder = '/scratch/lameeus/data/ghent_altar/altarpiece_close_up/finger/'
            self.image_clean = ImageStruct('clean', folder + 'hand_cleaned.tif')
            self.image_rgb = ImageStruct('before cleaning', folder + 'hand_rgb.tif')
            self.image_ir = ImageStruct('IR', folder + 'hand_ir.tif')

            folder = '/home/lameeus/data/ghent_altar/annotation/'
            self.image_annot = ImageStruct('annotations by restorer', folder + '19_annot_clean_big.tif')

            folder = '/home/lameeus/data/ghent_altar/output/'
            self.image_output = ImageStruct('demo result', folder + 'hand_big.tif')
        
    def get_input_images(self):
        return [self.image_clean, self.image_rgb, self.image_ir]
    
    def get_annot_images(self):
        return [self.image_annot]
    
    def get_output_images(self):
        return [self.image_output]
    

class GUIData(object):
    image_clean = ImageStruct('clean', '/home/lameeus/Pictures/Selection_008.png')
    image_annot = ImageStruct('annot', '/home/lameeus/Pictures/Selection_010.png')
    folder = '/home/lameeus/data/ghent_altar/input/'
    image_clean2 = ImageStruct('clean2', folder + '19_clean.tif')
    del(folder)
    set_struct = SetStruct('evangelist')
    
    names_sets = ['hand_small', 'hand_big']
    
    def __init__(self):
        ...
    
    def get_input_images(self):
        return self.set_struct.get_input_images()
    
    def get_annot_images(self):
        return self.set_struct.get_annot_images()
    
    def get_output_images(self):
        return self.set_struct.get_output_images()
    
    def get_names_sets(self):
        return self.names_sets
