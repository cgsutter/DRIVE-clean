import numpy as np
import tensorflow as tf  # using tf. functions where keras.ops previously was (bc using keras 2.11 not 3.x)
import keras
from keras.utils import register_keras_serializable  # CS
import tensorflow as tf
from tensorflow.math import digamma, lgamma


# def evidential_cat_loss(evi_coef, epoch_callback, class_weights=None):

#     def calc_kl(alpha):
#         beta = tf.ones(shape=(1, alpha.shape[1]), dtype=tf.float32)
#         S_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)
#         S_beta = tf.reduce_sum(beta, axis=1, keepdims=True)
#         lnB = lgamma(S_alpha) - tf.reduce_sum(lgamma(alpha), axis=1, keepdims=True)
#         lnB_uni = tf.reduce_sum(lgamma(beta), axis=1, keepdims=True) - lgamma(S_beta)
#         dg0 = digamma(S_alpha)
#         dg1 = digamma(alpha)
#         if class_weights is not None:
#             kl = (tf.reduce_sum(class_weights * (alpha - beta) * (dg1 - dg0), axis=1, keepdims=True)
#                   + lnB + lnB_uni)
#         else:
#             kl = (tf.reduce_sum((alpha - beta) * (dg1 - dg0), axis=1, keepdims=True)
#                   + lnB + lnB_uni)
#         return kl

#     # @keras.saving.register_keras_serializable()
#     @register_keras_serializable() #CS

#     def loss(y, y_pred):
#         current_epoch = epoch_callback.epoch_var
#         evidence = tf.nn.relu(y_pred)
#         alpha = evidence + 1
#         s = tf.reduce_sum(alpha, axis=1, keepdims=True)
#         m = alpha / s

#         if class_weights is not None:
#             a = tf.reduce_sum(class_weights * tf.square(y - m), axis=1, keepdims=True)
#             b = tf.reduce_sum(class_weights * alpha * (s - alpha) / (s * s * (s + 1)), axis=1, keepdims=True)
#         else:
#             a = tf.reduce_sum(tf.square(y - m), axis=1, keepdims=True)
#             b = tf.reduce_sum(alpha * (s - alpha) / (s * s * (s + 1)), axis=1, keepdims=True)

#         annealing_coef = tf.minimum(1.0, tf.cast(current_epoch, tf.float32))


@register_keras_serializable()

# backend = keras.backend.backend()

# if backend == "tensorflow":
#     try:
#         from tensorflow.math import digamma, lgamma
#     except ImportError:
#         print("Tensorflow not available")
# elif backend == "jax":
#     try:
#         from jax.scipy.special import digamma
#         from jax.lax import lgamma
#     except ImportError:
#         print("jax not available")
# elif backend == "torch":
#     try:
#         from torch.special import digamma
#         from torch import lgamma
#     except ImportError:
#         print("pytorch not available")

# @keras.saving.register_keras_serializable() #CS


def evidential_cat_loss(evi_coef, epoch_callback, class_weights=None):
    def calc_kl(alpha):
        beta = tf.ones(shape=(1, tf.shape(alpha)[1]), dtype=tf.float32)
        S_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)
        S_beta = tf.reduce_sum(beta, axis=1, keepdims=True)
        lnB = tf.math.lgamma(S_alpha) - tf.reduce_sum(
            tf.math.lgamma(alpha), axis=1, keepdims=True
        )
        lnB_uni = tf.reduce_sum(
            tf.math.lgamma(beta), axis=1, keepdims=True
        ) - tf.math.lgamma(S_beta)
        dg0 = tf.math.digamma(S_alpha)
        dg1 = tf.math.digamma(alpha)
        if class_weights is not None:
            kl = (
                tf.reduce_sum(
                    class_weights * (alpha - beta) * (dg1 - dg0), axis=1, keepdims=True
                )
                + lnB
                + lnB_uni
            )
        else:
            kl = (
                tf.reduce_sum((alpha - beta) * (dg1 - dg0), axis=1, keepdims=True)
                + lnB
                + lnB_uni
            )
        return kl

    @register_keras_serializable()
    def loss(y, y_pred):
        y = tf.cast(y, tf.float32)
        current_epoch = epoch_callback.epoch_var
        evidence = tf.nn.relu(y_pred)
        alpha = evidence + 1.0
        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        m = alpha / S

        if class_weights is not None:
            a = tf.reduce_sum(class_weights * tf.square(y - m), axis=1, keepdims=True)
            b = tf.reduce_sum(
                class_weights * alpha * (S - alpha) / (S**2 * (S + 1)),
                axis=1,
                keepdims=True,
            )
        else:
            a = tf.reduce_sum(tf.square(y - m), axis=1, keepdims=True)
            b = tf.reduce_sum(
                alpha * (S - alpha) / (S**2 * (S + 1)), axis=1, keepdims=True
            )

        annealing_coef = tf.minimum(1.0, tf.cast(current_epoch, tf.float32) / evi_coef)
        alpha_hat = y + (1 - y) * alpha
        c = annealing_coef * calc_kl(alpha_hat)
        c = tf.reduce_mean(c, axis=1)

        return tf.reduce_mean(a + b + c)

    return loss
