import tensorflow as tf
import numpy as np
import json
import math


class Manual:
    def __init__(self):
        self.useRaw = False
        pass

    def predict(self, x0, y0, vx, vy, h_w):
        xC = -1 if vx < 0 else 1
        yC = y0 + (xC - x0) * vy / vx / h_w

        return math.acos(math.cos(yC * math.pi)) / math.pi

    @staticmethod
    def predict_static(x0, y0, vx, vy, h_w):
        xC = -1 if vx < 0 else 1
        yC = y0 + (xC - x0) * vy / vx / h_w

        return math.acos(math.cos(yC * math.pi)) / math.pi

    def loadData(self):
        print("nothing to load [Manual]")

    def train(self, batchSize, epochs, verbose=1):
        print("no need to train [Manual]")

    def export(self):
        print("no need to export [Manual]")


class Neural_Network:
    def __init__(self):
        self.useRaw = False
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(4, input_shape=(4,),
                                  activation=tf.nn.elu),
            tf.keras.layers.Dense(4, activation=tf.nn.elu),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
        ])
        opt = tf.keras.optimizers.SGD(
            lr=0.5, momentum=0.6, nesterov=True)
        self.model.compile(
            loss='mean_squared_error',
            optimizer=opt
        )
        self.in__data = None
        self.out_data = None

    @staticmethod
    def loadFromBackup(bkp):
        raise Exception("not implemented")

    def loadData(self):
        with open("./data/data_num.json") as f:
            raw = json.load(f)['data']
            self.in__data = np.array(
                [[t['x'], t['y'], t['vx'], t['vy']] for t in raw])
            self.out_data = np.array(
                [[t['y_pred']] for t in raw])

    def train(self, batch_size_, epochs_, verbose_=1):
        if self.in__data is None or self.out_data is None:
            self.loadData()

        self.model.fit(
            x=self.in__data,
            y=self.out_data,
            batch_size=batch_size_,
            epochs=epochs_,
            verbose=verbose_
        )

    def predict(self, x0, y0, vx, vy, h_w=None):
        return self.model.predict(np.array([[x0, y0, vx, vy]]))[0][0]

    def export(self):
        raise Exception("not implemented")


class Deep_Learning:
    pass
