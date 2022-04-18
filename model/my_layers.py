import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense



class Positional_Encoding(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Positional_Encoding, self).__init__(**kwargs)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def build(self, input_shape):
        pass

    def call(self, inputs):
        input_shape = np.asarray(inputs.get_shape().as_list()[1:])
        position = input_shape[0]
        d_model = input_shape[1]
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)

        inputs *= tf.math.sqrt(tf.cast(d_model, tf.float32))
        inputs += pos_encoding[:, :position, :]
        return inputs


class p_w_ff(keras.layers.Layer):
    def __init__(self, **kwargs):
        # Nothing special to be done here
        super(p_w_ff, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x):
        input_shape = np.asarray(x.get_shape().as_list()[1:])
        return Dense(input_shape[1])(Dense(input_shape[0], activation='relu')(x))
