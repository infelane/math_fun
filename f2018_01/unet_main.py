import argparse
# from . import parser_net, builder_net
from f2018_01 import builder_net, train_net, data_sets
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random


def main():
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="* Network on *")
    parser.add_argument('--dataset', default='art',
                        help="The dataset")
    parser.add_argument('--epochs', default=50000, type=int)
    parser.add_argument('--lr', default=1e-3, type=float,
                        help="Initial learning rate")
    parser.add_argument('--decay', default=1e-6, type=float,
                        help="learning rate decay over EACH UPDATE.")
    # complex_unet, complex_unet_shift ...
    parser.add_argument('--model', default='complex_unet',
                        help="The model type")
    parser.add_argument('--save_dir', default='/home/lameeus/data/general/weights/dirty')     # where to save
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--input', default='all',
                        help='all or dirty: for ignoring the cleaned input')
    args = parser.parse_args()
    print(args)

    # load data
    if args.dataset == 'art':
        ext_tot = 26     # 6 for non-shift
        (x_train, y_train), (x_test, y_test), (x_val, y_val)= data_sets.load_art(ext_tot=ext_tot)
    elif args.dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = data_sets.load_mnist()
    else:
        raise ValueError('Unknown dataset')
    
    if args.input == 'all':
        ... # Nothing to do
    elif args.input == 'dirty':
        # remove clean
        def remove_clean(lst):
            return [lst[1], lst[2]]
        x_train = remove_clean(x_train)
        x_test = remove_clean(x_test)
        x_val = remove_clean(x_val)
    else:
        raise ValueError('Unknown input set')
    
    # choose the model
    if args.model == 'shallow_fcc':
        model = builder_net.shallow_fcc(x=x_train, y=y_train, units=20)
    elif args.model == 'simple':
        model = builder_net.simplest(x=x_train, y=y_train)
    elif args.model == 'shallow_cnn':
        model = builder_net.shallow_cnn(x=x_train, y=y_train, units=50)
    elif args.model == 'simple_cnn':
        model = builder_net.simplest_cnn(x=x_train, y=y_train)
    elif args.model == 'simple_unet':
        model = builder_net.simple_unet(x=x_train, y=y_train)
    elif args.model == 'simple_unet_shift':
        model = builder_net.simple_unet_shift(x=x_train, y=y_train)
    elif args.model == 'complex_unet':
        model = builder_net.complex_unet(x=x_train, y=y_train)
    elif args.model == 'complex_unet_shift':
        model = builder_net.complex_unet_shift(x=x_train, y=y_train)
    else:
        raise ValueError('Unknown model architecture')
        
    model.summary()

    if args.weights is not None:    # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train_net.train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        train_net.test(model=model, data=(x_val, y_val), args=args)


if __name__ == "__main__":
    main()
