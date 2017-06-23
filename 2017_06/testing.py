# # import numpy as np
# # import matplotlib.pyplot as plt
# #
# # class Formatter(object):
# #     def __init__(self, im):
# #         self.im = im
# #     def __call__(self, x, y):
# #         z = self.im.get_array()[int(y), int(x)]
# #         return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)
# #
# # data = np.random.random((10,10))
# #
# # fig, ax = plt.subplots()
# # im = ax.imshow(data, interpolation='none')
# # ax.format_coord = Formatter(im)
# # plt.show()
# #
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import mpldatacursor
# #
# # data = np.random.random((10,10))
# #
# # fig, ax = plt.subplots()
# # ax.imshow(data, interpolation='none')
# #
# # mpldatacursor.datacursor(hover=True, bbox=dict(alpha=1, fc='w'))
# # plt.show()
# #
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import mpldatacursor
# #
# # data = np.random.random((10,10))
# #
# # fig, ax = plt.subplots()
# # ax.imshow(data, interpolation='none', extent=[0, 1.5*np.pi, 0, np.pi])
# #
# # mpldatacursor.datacursor(hover=True, bbox=dict(alpha=1, fc='w'),
# #                          formatter='i, j = {i}, {j}\nz = {z:.02g}'.format)
# # plt.show()
#
# from PIL import Image
import pylab

#
# folder = '/scratch/lameeus/data/altarpiece_close_up/beard/'
# name =  '13IRRASS0001.tiff' # IRR full
# name =  '_8_panelXRR53.tif' #
# path_im = folder + name
#
# im = pylab.array(Image.open(path_im))
# pylab.imshow(im, interpolation = 'nearest', cmap = 'gray')
# print('Please click 3 points')
# x = pylab.ginput(3)
# print('you clicked:',x)
# pylab.show()

import rawpy
import imageio

path = 'image.raw'
raw = rawpy.imread(path)
rgb = raw.postprocess()
# imageio.imsave('default.tiff', rgb)