# For easy loading of whole network
"""
Default operations on network that are no part of the network
"""

# 3th party library
import os
import sys

# Own libraries
from f2017_01.tensorflow_folder import config_lamb
folder_loc = '/home/lameeus/Documents/Link to Python/2017_February/super_res_challenge'
cmd_subfolder = os.path.realpath(folder_loc)
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
from lambnet import network


class NetworkBase():
    def __init__(self, layers, flag):
        self.network_group = network.NetworkGroup(layers=layers, bool_residue=False)
        self.flag = flag
                
    def load_prev(self, bool = True):
        if bool:
            self.network_group.load_params(self.flag.checkpoint_dir)
        else:
            self.network_group.load_init()


# some examples
def base_lamb():
    # load settings
    layers = config_lamb.nn4()
    flag = config_lamb.FLAGS1()
        
    # build
    network_base = NetworkBase(layers, flag)
    network_base.load_prev(flag.load_prev)
    return network_base
    
