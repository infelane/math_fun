"""
For evaluation the network
"""
# TODO make it work

#3th party libraries
import numpy as np

from f2017_04 import net_builder


def eval1(net_base):
    params = net_base.network_group.params
    
    keys = sorted(params)
    for param in keys:
        # print(param)
        
        if param[0]== 'W':
            w_var = params[param]
            w_val = net_base.network_group.sess.run(w_var)
            # print(w_val)
            # print(np.shape(w_val))
            
            w_shape = np.shape(w_val)

            w_mean = np.mean(w_val)
            # print(w_mean)
            std = np.std(w_val)
            w_var = np.var(w_val)
            # print(std)
            # print(int(1/std))
            
            init = (2/(w_shape[0]*w_shape[1]*w_shape[2] + w_shape[0]*w_shape[1]*w_shape[3]) )**-1
            
            print('{}\t{}\t{}\t{}\t{}'.format(param, w_shape, w_var, int(1/w_var), int(init)))

        if param[0] == 'b':
            w_var = params[param]
            w_val = net_base.network_group.sess.run(w_var)
            # print(w_val)
            # print(np.shape(w_val))
    
            w_shape = np.shape(w_val)
    
            w_mean = np.mean(w_val)
            # print(w_mean)
            std = np.std(w_val)
            w_var = np.var(w_val)
            # print(std)
            # print(int(1/std))
    
            print('{}\t{}\t{}\t{}\t{}\t{}'.format(param, w_shape, w_mean, std, w_var, int(1 / w_var)))

def main():
    net_base = net_builder.base_lamb()
    # load the latest weights
    net_base.load_prev()
        
    eval1(net_base)
        

if __name__ == '__main__':
    main()
