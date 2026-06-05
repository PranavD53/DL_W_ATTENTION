import tensorflow as tf
from tensorflow.keras.layers import Layer

class AttentionLayer(Layer):

    def build(self, input_shape):

        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )

        self.b = self.add_weight(
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):

        score = tf.nn.tanh(
            tf.tensordot(
                inputs,
                self.W,
                axes=1
            ) + self.b
        )

        weights = tf.nn.softmax(
            score,
            axis=1
        )

        context = tf.reduce_sum(
            weights * inputs,
            axis=1
        )

        return context