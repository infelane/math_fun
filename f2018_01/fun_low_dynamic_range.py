import matplotlib.pyplot as plt
import numpy as np

def main():
    file = "Lenna.png"
    file = 'ruth_laurens.jpg'
    im = plt.imread(file)
    
    a = np.max(im)
    print(a)
    
    im = im/a
    
    shape = np.shape(im)
    random = np.random.uniform(0., 1., size=shape)
    
    bool_im = np.greater_equal(im, random)
    
    new_im = np.zeros(shape)
    new_im[bool_im] = 1.
    
    plt.imsave("Lenna_LDR.png", new_im)
    
    plt.subplot(1, 2, 1)
    plt.imshow(im)
    plt.subplot(1, 2, 2)
    plt.imshow(new_im)
    plt.show()


if __name__ == '__main__':
    main()
