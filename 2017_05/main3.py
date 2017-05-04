# should be cleaner than main2
# probs will convert everything to keras

import keras
import lambnet

import os, sys

#3th party
folder_loc = '/ipi/private/lameeus/private_Documents/python/2017_January/tensorflow_folder'
cmd_subfolder = os.path.realpath(folder_loc)
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import config_lamb


def main():
    layers = config_lamb.nn4()
    
    lambnet.block_builder.foo(layers)
    # lambnet.
    
    print(layers.layer_types)
    

if __name__ == '__main__':
    main()
    