import numpy as np
import argparse

from f2018_01 import data_sets, builder_net, train_net

def get_args():

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="DESCRIPTION")
    parser.add_argument('--decay', default=0., type=float,
                        help="learning rate decay over EACH UPDATE. 1e-3 is halving after 700ish epochs")
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=1e-3, type=float,
                        help="Initial learning rate")
    parser.add_argument('--save_dir', default='/home/lameeus/data/general/weights/2018IEEE_challenge')     # where to save
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)
    
    return args

def main():
    args = get_args()
    
    if 1: # data
        (x_train, y_train), (x_test, y_test) = data_sets.load_multi()
        print(np.shape(x_train))
        print(np.shape(y_test))

    model = builder_net.shallow_fcc(x=x_train, y=y_train, units=20)

    train_net.train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args, bool_generator=False)


if __name__ == '__main__':
    main()
