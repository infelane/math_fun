from link_to_soliton.paint_tools.image_tools import path2im, save_im
import matplotlib.pyplot as plt
import numpy as np


def plot3(im1, im2, im3):
    plt.subplot(1, 3, 1)
    plt.imshow(im1)
    plt.subplot(1, 3, 2)
    plt.imshow(im2)
    plt.subplot(1, 3, 3)
    plt.imshow(im3)


def main():
    #
    folder = '/home/lameeus/data/ghent_altar/input/'
    path = folder + '19_clean.tif'
    im_clean = path2im(path)
    
    if 0:
        folder = '/home/lameeus/data/ghent_altar/annotation/'
        # path = folder + '19_annot_clean_big_cyan.tif'
        path = folder + '19_annot_clean_big_red_only.tif'
    else:
        folder = '/home/lameeus/data/ghent_altar/input/'
        path = folder + '19_annot_clean_big.tif'
    im_annot = path2im(path)

    folder= '/home/lameeus/data/ghent_altar/classification/unet/'
    path = folder + 'pred_certainty_hand.png'
    im_pred = path2im(path)
    
    print(np.shape(im_clean))
    print(np.shape(im_annot))
    print(np.shape(im_pred))
    
    if 0:
        plot3(im_clean, im_annot, im_pred)
        plt.show()

    w = 200
    point = [[4360, 3150], [5750, 4150], [600, 3300], [4400, 4350]]
    for point_i in point:
        plt.figure()
        im_clean_crop = im_clean[point_i[0]:point_i[0]+w, point_i[1]:point_i[1]+w ]
        im_annot_crop = im_annot[point_i[0]:point_i[0] + w, point_i[1]:point_i[1] + w]
        im_pred_crop = im_pred[point_i[0]:point_i[0] + w, point_i[1]:point_i[1] + w]

        plot3(im_clean_crop, im_annot_crop, im_pred_crop)
        
        folder = '/home/lameeus/data/ghent_altar/output/for_plots/'
        save_im(im_clean_crop, path=folder + 'clean_{}_{}.png'.format(point_i[0], point_i[1]))
        save_im(im_annot_crop, path=folder + 'annot_{}_{}.png'.format(point_i[0], point_i[1]))

        
        cmap = plt.cm.jet
        path = folder + 'pred_{}_{}.png'.format(point_i[0], point_i[1])
        if 0:
            save_im(im_pred_crop, path=path)
        else:
            plt.imsave(path, im_pred_crop, cmap=cmap)
        
if __name__ == '__main__':
    main()