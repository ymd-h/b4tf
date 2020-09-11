from typing import Iterable

import tensorflow as tf
from tensorflow.python.framework import tensor_shape

import tensorflow_probability as tfp


class PBPLayer(tf.keras.layers.Layer):
    """
    Layer for Probabilistic Backpropagation
    """
    def __init__(self,units: int,
                 alpha : tf.Variable,
                 beta: tf.Variable,
                 *args,**kwargs):
        """
        Initialize PBP layer

        Parameters
        ----------
        units: int
           Number of units in layer. (Output shape)
        alpha : tf.Variable
           Model wide parameter alpha for Gamma(alpha, beta)
        beta : tf.Variable
           Model wide parameter beta for Gamma(alpha, beta)
        """
        super().__init__(*args,**kwargs)
        self.units = units
        self.alpha = alpha
        self.beta = beta

    def build(self,input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `PBPLayer` '
                             'should be defined. Found `None`.')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        self.inv_sqrtV1 = 1.0 / tf.math.sqrt(last_dim + 1)
        self.inv_V1 = tf.math.square(inv_sqrtV1)


        self.kernel_m = self.add_weight("kernel_mean",
                                        shape=[last_dim,self.units],
                                        initializer=,
                                        dtype=self.dtype,
                                        trainable=True)
        self.kernel_v = self.add_weight("kernel_variance",
                                        shape=[last_dim,self.units],
                                        initializer=,
                                        dtype=self.dtype,
                                        trainable=True)
        self.bias_m = self.add_weight("bias_mean",
                                      shape=[self.units,],
                                      initializer=,
                                      dtype=self.dtype,
                                      trainable=True)
        self.bias_v = self.add_weight("bias_variance",
                                      shape=[self.units,],
                                      initializer=,
                                      dtype=self.dtype,
                                      trainable=True)
        self.kernel_fn = tfp.distributions.
        self.built = True

    @tf.function
    def _Z(self,y: tf.Tensor,
           alpha: tf.Tensor, beta: tf.Tensor, m: tf.Tensor, v: tf.Tensor):
        std = tf.math.sqrt(tf.math.maximum(beta/(alpha-1) + v,tf.zeros_like(v)))
        N = tfp.distributions.Normal(loc=m, scale=std)
        return N.prob(y)

    def fit(self,x,y):
        pass

    def call(self,*args,**kwargs):
        pass

    def get_config(self):
        return {**super().get_config(),}



class PBP:
    """
    Probabilistic Backpropagation

    References
    ----------
    J. M. Hernández-Lobato and R. P. Adams,
    "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks",
    arXiv 1502.05336, 2015
    """
    def __init__(self,units):
        pass

    def prob(self,x):
        pass
