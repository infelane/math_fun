# Preprocessing step

from skimage import color
import numpy as np

# transforming to L*a*b* space
def func1(data):
    x = np.empty(shape = np.shape(data))

    x[:, :, :, 0:3] = color.rgb2lab(data[..., 0:3])
        
    x[:, :, :, 3:6] = color.rgb2lab(data[..., 3:6])
    x[:, :, :, 6] = data[..., 6]
        
    return x

def rgb2lab(data):
    x = color.rgb2lab(data)
    return x

def lab2rgb(data):
    return color.lab2rgb(data)
    