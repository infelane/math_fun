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
    
    names_sets = ['hand_small', 'hand_big', 'zach_small', 'zach_big']
    
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
    
    
def inpaint_set_book():
    path_orig = '/home/lameeus/data/ghent_altar/roman/input/16_VIS_reg.tif'
    path_map = '/home/lameeus/data/ghent_altar/roman/input/crack_cnn.png'
    path_result = '/home/lameeus/data/ghent_altar/roman/stitch/inpainting_full_v4.png'
    return InpaintingSet('book', path_orig, path_map, path_result)


def inpaint_set_zach():
    folder = '/home/lameeus/data/ghent_altar/inpainting/detection_and_inpainting/Zachary/'
    path_orig = folder + 'cut_ori_3.png'
    path_map = folder + 'src_i3_with.png'
    path_result = folder + 'src_inpainted_face.png'
    path_restored = '/home/lameeus/data/ghent_altar/bart_devolder/02/x103112l.tif'
    return InpaintingSet('zachary', path_orig, path_map, path_result, path_restored)


def inpaint_set_hand():
    folder = '/home/lameeus/data/ghent_altar/inpainting/detection_and_inpainting/John_the_Evangelist/'
    path_orig = folder + 'original_rgb_hand.tif'
    path_map = folder + 'src_i3_with.png'
    path_result = folder + 'src_inpainted_whole.png'
    path_restored ='/home/lameeus/data/ghent_altar/bart_devolder/02/x103692l.tif'
    return InpaintingSet('hand', path_orig, path_map, path_result, path_restored)

    
class InpaintingSet(object):
    def __init__(self, name, path_orig, path_map, path_result, path_restored = None):
        self.name = name
        self.path_orig = path_orig
        self.path_map = path_map
        self.path_result = path_result
        self.path_restored = path_restored
        
    def get_orig(self):
        return image_tools.path2im(self.path_orig)
    
    def get_map(self):
        return image_tools.path2im(self.path_map)
    
    def get_result(self):
        return image_tools.path2im(self.path_result)
    
    def get_restored(self):
        if self.path_restored is None:
            return None
        else:
            return image_tools.path2im(self.path_restored)
    
    
class InpaintingData(object):
    sets = [inpaint_set_book(), inpaint_set_hand(), inpaint_set_zach()]
    
    def get_sets(self):
        return self.sets
