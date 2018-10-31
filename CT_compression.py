import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinv

def main():
    im = plt.imread('/home/lameeus/Pictures/elephant.jpg')
    
    # subsample
    rate = 10
    im = im[::rate, ::rate, :]
    
    # grey
    im = np.mean(im, axis=2)
    
    if 0:
        plt.figure()
        plt.imshow(im)
        plt.show()
    
    shape = np.shape(im)
    h, w = shape
    
    flat = np.reshape(im, (h*w,))
    
    # # sum each column
    # T1 = np.zeros((h*w, h))
    # for i_h in range(h):
    #     a = np.zeros(shape)
    #     a[i_h, :] = 1
    #
    #     T1[..., i_h] = np.reshape(a, newshape=(h*w, ))
    #
    # # sum each row
    # T2 = np.zeros((h * w, w))
    # for i_w in range(w):
    #     a = np.zeros(shape)
    #     a[:, i_w] = 1
    #     T2[..., i_w] = np.reshape(a, newshape=(h * w,))
    
    # sum each column
    T1 = np.zeros((h, w, w))
    for i_h in range(h):
        T1[i_h, :, i_h] = 1
    T1 = T1.reshape((h*w, w))
        
    # sum each row
    T2 = np.zeros((h, w, w))
    for i_w in range(w):
        T2[:, i_w, i_w] = 1
    T2 = T2.reshape((h*w, w))

    sum1 = np.matmul(flat, T1)
    sum2 = np.matmul(flat, T2)
    
    T_tot = np.concatenate([T1, T2], axis=-1)
    
    sum_tot = np.concatenate([sum1, sum2], axis=-1)

    T_tot_pinv = np.linalg.pinv(T_tot)
    
    flat_reconstruct = np.matmul(sum_tot, T_tot_pinv)
    im_reconstruct = np.reshape(flat_reconstruct, (h, w))
    
    
if __name__ == '__main__':
    main()
