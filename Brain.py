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

    def predict_raw(self, pixels):
        raise NotImplementedError("method not implemented")

    def load_data(self):
        print("nothing to load [Manual]")

    def train(self, batch_size_, epochs_, verbose_=1):
        print("no need to train [Manual]")

    def export(self):
        print("no need to export [Manual]")


class Neural_Network:
    def __init__(self):
        self.useRaw = False
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(4, input_shape=(4,),
                                  activation=tf.keras.activations.elu),
            tf.keras.layers.Dense(4, activation=tf.keras.activations.elu),
            tf.keras.layers.Dense(
                1, activation=tf.keras.activations.sigmoid),
        ])
        opt = tf.keras.optimizers.SGD(
            lr=0.5, momentum=0.6, nesterov=True
        )
        self.model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=opt
        )
        self.in__data = None
        self.out_data = None

    @staticmethod
    def loadFromBackup(bkp):
        raise NotImplementedError("method not implemented")

    def load_data(self, address="./data/data_num.json"):
        with open(address, "r") as f:
            raw = json.load(f)['data']
            self.in__data = np.array(
                [[t['x'], t['y'], t['vx'], t['vy']] for t in raw])
            self.out_data = np.array([[t['y_pred']] for t in raw])

    def train(self, batch_size_, epochs_, verbose_=1):
        if self.in__data is None or self.out_data is None:
            self.load_data()

        self.model.fit(
            x=self.in__data,
            y=self.out_data,
            batch_size=batch_size_,
            epochs=epochs_,
            verbose=verbose_
        )

    def predict(self, x0, y0, vx, vy, h_w=None):
        return self.model.predict(np.array([[x0, y0, vx, vy]]))[0][0]

    def predict_raw(self, pixels):
        raise NotImplementedError("method not implemented")

    def export(self):
        raise NotImplementedError("method not implemented")


class Deep_Learning:
    def __init__(self):
        self.useRaw = True
        self.inputShape = (None, None)

        self.model = None

        self.in__data = None
        self.out_data = None

    def setup_model(self):
        if self.inputShape is None:
            # raise ValueError("input size is not defined try loading data")
            self.inputShape = (60, 90)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                input_shape=(60, 90, 1),
                filters=3,
                kernel_size=10
            ),
            tf.keras.layers.MaxPool2D(),
            # tf.keras.layers.Conv2D(
            #     filters=3,
            #     kernel_size=10
            # ),
            # tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=5, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(
                units=1, activation=tf.keras.activations.sigmoid)
        ])
        opt = tf.keras.optimizers.SGD(
            lr=0.5, momentum=0.6, nesterov=True)
        self.model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=opt
        )

    @staticmethod
    def loadFromBackup(bkp):
        raise NotImplementedError("method not implemented")

    def load_data(self, address="./data/data_frame.serialized"):
        f = open(address, "r")
        n_entries = int(f.read(4))
        entries_h = int(f.read(4))
        entries_w = int(f.read(4))
        content = f.read().split(';')[:-1]  # eclude last ''

        if not len(content) == n_entries:
            raise ValueError("entries number is nonsense")

        in__raw = []
        out_raw = []

        for c in content:
            values = c.split(',')

            out_raw.append([float("0." + values[-1])])

            values = values[:-1]
            if not len(values) == entries_h * entries_w:
                raise ValueError("frame dimension is nonsense")
            in__raw.append(
                np.array([[float(values[i * entries_w + j]) / 255.0
                           for j in range(entries_w)] for i in range(entries_h)])
            )

        f.close()

        self.in__data = np.expand_dims(np.array(in__raw), axis=3)
        self.out_data = np.array(out_raw)
        self.inputShape = (entries_h, entries_w)

        if True:
            print("decoded: ")
            print(n_entries)
            print(entries_h)
            print(entries_w)
            print(self.in__data.shape)
            print(self.out_data.shape)

        self.setup_model()

    def train(self, batch_size_, epochs_, verbose_=1):
        if self.in__data is None or self.out_data is None:
            self.load_data()

        self.model.fit(
            x=self.in__data,
            y=self.out_data,
            batch_size=batch_size_,
            epochs=epochs_,
            verbose=verbose_
        )

    def predict(self, x0, y0, vx, vy, h_w=None):
        raise NotImplementedError("method not implemented")

    def predict_raw(self, pixels):
        """ expected shape for pixels : (1, 60, 90, 1)
        """
        return self.model.predict(pixels)

    def export(self):
        raise NotImplementedError("method not implemented")
