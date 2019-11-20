import tensorflow as tf
import numpy as np
import Brain


def ask(s):
    i = None
    while i not in ['y', 'n']:
        if i is not None:
            print("please, use y/n")
        i = input(s).lower()
    return i


def train_DL():
    DL = Brain.Deep_Learning()
    DL.load_data()
    DL.setup_model()

    print("loss b.t.: {}".format(DL.model.evaluate(DL.in__data, DL.out_data)))

    while ask('train? [y/n] ') == 'y':
        DL.train(50, 50, verbose_=0)
        print("loss a.t.: {}".format(DL.model.evaluate(DL.in__data, DL.out_data)))

    if ask('save? [y/n] ') == 'y':
        DL.export()
        print('saved !')


def train_NN():
    NN = Brain.Neural_Network()
    NN.load_data()
    NN.setup_model()

    print("initial loss: {}".format(NN.model.evaluate(NN.in__data, NN.out_data)))

    while ask('train? [y/n] ') == 'y':
        NN.train(500, 2000, verbose_=0)
        print("loss after training: {}".format(
            NN.model.evaluate(NN.in__data, NN.out_data)))

    if ask('save? [y/n] ') == 'y':
        NN.export()
        print('saved !')


if __name__ == "__main__":
    train_DL()
