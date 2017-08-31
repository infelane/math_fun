""" the framework
"""

import numpy as np
import time
import matplotlib.pyplot as plt

from f2017_08.hsi import tools_datasets, tools_data, nn, tools_plot
from maus.paint_tools import image_tools


def main():
    t0 = time.time()
    img_norm = tools_datasets.hsi_processed()

    img_norm = img_norm[::, ::, :]

    from f2017_08 import main_multimodal
    hsi_data = main_multimodal.HsiData(load = True)

    rgb = hsi_data.to_rgb(img_norm)
    plt.imshow(rgb)
    # plt.show()
    
    data = tools_data.Data(img_norm)
    
    x = data.img_to_x(img_norm)

    # import maus.paint_tools.image_tools
    # mask = maus.paint_tools.image_tools.path2im('/home/lameeus/data/hsi/mask.png')

    mask = tools_datasets.hsi_mask()

    print(mask)
    print(np.shape(mask))

    x_train = data.img_mask_to_x(img_norm, mask)
    
    print(np.shape(x_train))
    
    # x_small = x[0:10000, ...]
    
    print(np.shape(x))
    
    # print(np.max(img_norm))
    print(np.shape(img_norm))

    auto_encoder = nn.AutoEncoder()
    auto_encoder.load()
    
    epochs = 20
    if epochs:
        auto_encoder.train(x_train, epochs = epochs, save = True)
    
    y = auto_encoder.predict(x)
    
    code = auto_encoder.predict_code(x)

    auto_encoder.stop()
    
    y_img = data.y_to_img(y)
    y_rgb = hsi_data.to_rgb(y_img)
    
    plt.figure()
    tools_plot.imshow(y_rgb, title='auto encoded')
    
    if 1:
        image_tools.save_im(y_rgb, '/ipi/research/lameeus/data/hsi/y_rgb.png')

    code_img = data.y_to_img(code)

    rgb = tools_plot.n_to_rgb(code_img)
    tools_plot.imshow(rgb, mask = mask, title = 'arg(p_max)')
    
    # rgb = tools_plot.n_to_rgb(code_img, with_lum=True)
    # plt.figure()
    # plt.imshow(rgb)

    # rgb = tools_plot.n_to_rgb(code_img, with_sat=True)
    # plt.figure()
    # plt.imshow(rgb)
    

    rgb = tools_plot.n_to_rgb(code_img, with_sat=True, with_lum=True)
    tools_plot.imshow(rgb, title = 'arg(p_max) and p_max')
    
    rgb = tools_plot.n_to_rgb(code_img, with_col = False, with_sat=True, with_lum=True)
    tools_plot.imshow(rgb, mask = mask, title = 'p_max')
    
    code_rgb = code_img[:,:,0:3]
    min = np.min(code_rgb)
    max = np.max(code_rgb)
    code_rgb = (code_rgb-min)/(max - min)
    
    plt.figure()
    plt.imshow(code_rgb)
    
    plt.show()
    
    print(np.shape(y))
    print(np.shape(y_img))
    
    t1 = time.time()
    
    total = t1 - t0
    print(total)

if __name__ == '__main__':
    main()