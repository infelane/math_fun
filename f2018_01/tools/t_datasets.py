import numpy as np
import spectral.io.envi as envi


def hsi_raw(folder, name_header, name_image):
    """
    hyperspectral image dataset from multispectral data
    :return:
    """

    lib = envi.open(folder + name_header, image=folder + name_image)
    
    img = np.array(lib[:, :, :])
    
    return img

def to_categorical_0_unknown(y):
    """Adjustment of keras.utils.to_categorical
    MAKE SURE THAT THE LAST DIM IS NON ZERO
    """
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    
    shape = np.shape(y)

    num_classes = np.max(y)    # with 0 meaning nothing, the number of classes is from #[1 : n]
    
    assert shape[-1] != 1

    shape_new = [a for a in shape] + [num_classes]
    y_new = np.zeros(shape_new, dtype=int)

    for i in range(1, num_classes+1): # ignore 0
        y_new[np.equal(y, i), i-1] = 1
    
    return y_new