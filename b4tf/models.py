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
        pi = tf.math.atan(tf.constant(1.0)) * 4
        self.log_inv_sqrt2pi = -0.5*tf.math.log(2.0*pi)
        self.built = True

    @tf.function
    def _logZ(self,y: tf.Tensor,
              alpha: tf.Tensor, beta: tf.Tensor, m: tf.Tensor, v: tf.Tensor):
        """
        Log of Partition Function

        Parameters
        ----------
        y : tf.Tensor
            Observed value
        alpha : tf.Tensor
            Parameter for Gamma(alpha, beta)
        beta : tf.Tensor
            Parameter for Gamma(alpha, beta)
        m : tf.Tensor
            Mean of N(m,v)
        v : tf.Tensor
            Variance of N(m,v)
        """
        std = tf.math.sqrt(tf.math.maximum(beta/(alpha-1) + v,
                                           tf.zeros_like(v)+1e-12))
        y_hat = (y - m)/std
        return tf.reduce_sum(tf.math.square(y_hat)+self.log_inv_sqrt2pi)

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
    def __init__(self,units: Iterable[int],*,
                 input_shape: Iterable[int]=(1,)):
        """
        Initialize PBP model

        Parameters
        ----------
        units : Iterable[int]
            Numbers of hidden units and outputs
        input_shape : Iterable[int], optional
            Input shape for PBP model. The default value is `(1,)`
        """
        self.alpha_gamma  = tf.Variable(1.0,trainable=True)
        self.alpha_lambda = tf.Variable(1.0,trainable=True)
        self.beta_gamma   = tf.Variable(0.0,trainable=True)
        self.beta_lambda  = tf.Variable(0.0,trainable=True)


        D = tf.keras.layers.Dense
        S = tf.keras.layers.Sequential

        self.input_shape = input_shape
        self.layers = S([[D(units[0],activation="relu",input_shape=input_shape)] +
                         [D(units[i],activation="relu") for i in units[1:-1]] +
                         [D(units[-1])]])

    def _Z(self):
        pass


    def prob(self,x):
        pass

    def __call__(self,x):
        x = tf.convert_to_tensor(x,shape=(-1,*self.input_shape))
        return self._call(x)

    @tf.function
    def _call(self,x:tf.Tensor):
        return self.layers(x)
