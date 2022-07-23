import tensorflow as tf

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras

class DqnAgent(object):
    """docstring for DqnAgent"""
    def __init__(self, s_size, a_size):
        super(DqnAgent, self).__init__()
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(s_size, input_shape=(s_size,), activation='relu'))
        self.model.add(keras.layers.Dense(a_size, activation='softmax'))
        self.model.compile(optimizer=tf.optimizers.Adam(0.7),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def get_action(self, state):
        return np.argmax(self.model.predict(state)[0])

    def predict(self, next_state):
        return self.model.predict(next_state)[0]

    def fit(self, state, target, action):
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)