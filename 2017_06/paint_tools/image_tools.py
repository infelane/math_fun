import tifffile as tiff
import numpy as np
import scipy


def path2im(path):
    # scipy.ndimage.imread(path) # DOES not work for 16 bit tif
    # Image.open(path_i) # does not open with 16 bit tif
    im = tiff.imread(path)  # colors stay correct
    # im = cv2.imread(path, -1)  # wrong color conversion (probs BGR), also bit slower?
    
    val_max = np.max(im)
    if val_max < 1:
        im = im  # already float
    elif val_max <= 255:
        im = im / 255.
    elif val_max <= 65535:
        im = im / 65536.
    
    return im

def save_im(array, path):
    # normalization between 0 and 1 is important!
    scipy.misc.toimage(array, cmin=0.0, cmax=1.0).save(path)

