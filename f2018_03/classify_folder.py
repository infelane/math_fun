from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os

from link_to_soliton.paint_tools import image_tools
from f2018_03 import test_models
from f2017_08.hsi import tools_data


class Classifier():
    def __init__(self):
        self.name = 'clean_no_spatial'
        
        # self.model = test_models.simplest2()
        
        self.model = test_models.trained_net()
    
    def predict(self, files):
        img_clean = files['clean']
        
        w_out = 10
 
        # from f2017_09 import main_lamb

        # dict_data = main_lamb.MainData(set='zach_big', w=w)
        
        # img_clean = dict_data.get_img_clean()
        # img_rgb = dict_data.get_img_rgb()
        # img_ir = dict_data.get_img_ir()

        data = tools_data.Data(img_clean, w=w_out)

        ext_double = 26
        ext_zoom = (ext_double) // 2
        ext_tuple = (ext_zoom, ext_double - ext_zoom)

        x_clean = data.img_to_x2(img_clean, ext=ext_tuple)

        x_in = x_clean

        y_pred = self.model.predict(x_in)

        y_img_pred = data.y_to_img2(y_pred)

        def get_pred_rgb(img_clean, pred_img):
            cyan = [0, 1, 1]
            pred_rgb = np.copy(img_clean)
            pred_rgb[pred_img[:, :, 1] > 0.5, :] = cyan
            return pred_rgb

        segmentation_rgb = get_pred_rgb(img_clean, y_img_pred)

        # plt.imsave('input.png', img_clean)
        # plt.imsave('prediction.png', segmentation_rgb)

        # from link_to_soliton.paint_tools.image_tools import save_im
        grey_loss = y_img_pred[..., 1]
        # # grey = np.stack([grey_single]*3, axis=2)
        # # save_im(grey, 'pred_certainty.png')
        # save_im(grey_single, 'pred_certainty.png')
        
        if 0:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(grey_loss)
            plt.subplot(1, 2, 2)
            plt.imshow(segmentation_rgb)
            plt.show()
        
        return y_img_pred, grey_loss, segmentation_rgb


def main():
    base_folder_path = '/home/lameeus/data/ghent_altar/input/hierarchy/'
    
    classifier = Classifier()
    
    for sub_folder_path in glob(base_folder_path+"*/"):
        sub_folder = sub_folder_path[len(base_folder_path):]
        folder_output = '/home/lameeus/data/ghent_altar/output/hierarchy/' + sub_folder \
                        + classifier.name + '/'
        
        path_save_pred_loss = folder_output + 'y_pred_loss.tif'
        path_save_pred_clean = folder_output + 'y_pred_clean.tif'
        
        if not (os.path.exists(path_save_pred_loss) or os.path.exists(path_save_pred_clean)):
            files = {}
            for sub_file in glob(sub_folder_path+"*.*"):
                
                img = image_tools.path2im(sub_file)
                i_start = len(sub_folder_path)
                name = sub_file[i_start:]
                i_point = name.find('.')
                name = name[:i_point]
                files.update({name:img})
            
            print('available images: {}'.format(files.keys()))
    
            _, grey_loss, segmentation_rgb = classifier.predict(files)
            
      
            
            
            print(folder_output)
    
            if not os.path.exists(folder_output):
                os.makedirs(folder_output)
                
            image_tools.save_im(grey_loss, path=path_save_pred_loss)
            image_tools.save_im(segmentation_rgb, path=path_save_pred_clean)


if __name__ == '__main__':
    main()
