import tensorflow as tf  # using tf. functions where keras.ops previously was (bc using keras 2.11 not 3.x)
import keras
import keras.layers as layers
from keras.utils import register_keras_serializable  # CS


# @keras.saving.register_keras_serializable() #CS
@register_keras_serializable()
class DenseNormalGamma(layers.Layer):
    """Implements dense output layer for a deep evidential regression model.
    Reference: https://www.mit.edu/~amini/pubs/pdf/deep-evidential-regression.pdf
    Source: https://github.com/aamini/evidential-deep-learning
    """

    NUM_OUTPUT_PARAMS = 4

    def __init__(
        self,
        units: int,
        spectral_normalization: bool = False,
        eps: float = 1e-12,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = int(units)
        if spectral_normalization:
            self.dense = layers.SpectralNormalization(
                layers.Dense(
                    DenseNormalGamma.NUM_OUTPUT_PARAMS * self.units, activation=None
                )
            )
        else:
            self.dense = layers.Dense(
                DenseNormalGamma.NUM_OUTPUT_PARAMS * self.units, activation=None
            )
        self.eps = eps

    def evidence(self, x):
        return tf.maximum(tf.nn.softplus(x), self.eps)

    def call(self, x):
        output = self.dense(x)
        mu, logv, logalpha, logbeta = tf.split(output, 4, axis=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return tf.concat([mu, v, alpha, beta], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], DenseNormalGamma.NUM_OUTPUT_PARAMS * self.units

    def get_config(self):
        base_config = super(DenseNormalGamma, self).get_config()
        base_config["units"] = self.units
        return base_config


# @keras.saving.register_keras_serializable() #CS
@register_keras_serializable()
class DenseNormal(layers.Layer):
    """Dense output layer for a Gaussian distribution regression neural network."""

    NUM_OUTPUT_PARAMS = 2

    def __init__(
        self,
        units: int,
        spectral_normalization: bool = False,
        eps: float = 1e-12,
        output_activation="sigmoid",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = int(units)
        self.output_activation = output_activation
        if spectral_normalization:
            self.dense = layers.SpectralNormalization(
                layers.Dense(
                    DenseNormal.NUM_OUTPUT_PARAMS * self.units,
                    activation=self.output_activation,
                )
            )
        else:
            self.dense = layers.Dense(
                DenseNormal.NUM_OUTPUT_PARAMS * self.units,
                activation=self.output_activation,
            )
        self.eps = eps

    def call(self, x):
        output = self.dense(x)
        output = tf.maximum(output, self.eps)
        mu, sigma = tf.split(output, 2, axis=-1)
        return tf.concat([mu, sigma], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], DenseNormal.NUM_OUTPUT_PARAMS * self.units

    def get_config(self):
        base_config = super(DenseNormal, self).get_config()
        base_config["units"] = self.units
        base_config["eps"] = self.eps
        base_config["output_activation"] = self.output_activation
        return base_config
