"""
small script to give me the plot data I want for paper
"""

import numpy as np
import glob
import matplotlib.pyplot as plt


def main():
    general_filenames = 'performance/*.csv'
    
    # fig1 = plt.figure(1)
    
    for file in glob.glob(general_filenames):
        # fig = plt.figure()
        print(file)
        my_data = np.genfromtxt(file, delimiter=',', skip_header=1) # watch out for header!
        
        print(np.shape(my_data))
        
        t = my_data[:, 1]
        y = my_data[:, 2]
        
        def smoother(a):
            # if np.isnan(a[0]):
            #     a[0] = 2*a[1] - a[0]# quick fix (linear interpolation)
                
            shape = np.shape(a)
            b = np.zeros(np.shape(a))
            n = shape[0]
            
            ext = 10
            
            interpolator_func = np.zeros((2*ext+1))
            interpolator_func[ext+1] = ext+1
            for i in range(ext):
                interpolator_func[i] = i+1
                interpolator_func[-i-1] = i+1
            
            for i in range(-ext, ext+1):  # include 2
                i_start = max(0, i)
                i_start_inverse = max(0, -i)
                b[i_start:n-i_start_inverse] += interpolator_func[ext + i]*a[i_start_inverse:n-i_start]

            averager = np.ones((n, )) * np.sum(interpolator_func) #2*ext+1
            for i in range(ext):
                averager[i] = np.sum(interpolator_func[0:ext + i+1]) # ext + 1 + i
                averager[-i-1] = np.sum(interpolator_func[0:ext + i+1])
                
            return b/averager
            
        # plt.plot(t, y, '.')

        label = ''
        
        if 'val' in file:
            label += 'val'
        else:
            label += 'train'
    
        if 'jaccard' in file:
            label += '_jaccard'
        else:
            label += 'cross_entropy'
        
        plt.plot(t, smoother(y), '--', label=label)
    
    plt.legend()
    plt.show()
    
    
    
    #
    # filename = 'performance/'
    # my_data = genfromtxt('my_file.csv', delimiter=',')
    
    
    
    
if __name__ == '__main__':
    main()