from typing import Iterable, Union

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from .base import ModelBase

__all__ = ["MCBN"]


class MCBN(ModelBase):
    """
    Monte Carlo Batch Normalization

    References
    ----------
    M. Teye et al.,
    "Bayesian Uncertainty Estimation for Batch Normalized Deep Networks",
    arXiv 1802.06455, 2018
    """
    BN_class = tf.keras.layers.BatchNormalization

    def __init__(self,units: Iterable[int],
                 noise_variance: float,*,
                 input_shape: Iterable[int]=(1,),
                 dtype: Union[tf.dtypes.DType,np.dtype,str]=tf.float32):
        """
        Initialize MCBN

        Parameters
        ----------
        units : Iterable[int]
            Number of units at hidden layers and output layer.
        noise_variance : float
            Variance of observation noise. (Hyper parameter)

        Raises
        ------
        ValueError
            If `noise_variance` is negative value
        """
        super().__init__(dtype, input_shape,units[-1])

        S = tf.keras.Sequential
        D = tf.keras.layers.Dense
        B = tf.keras.layers.BatchNormalization
        A = tf.keras.layers.Activation

        # Input Layer
        layers = [D(units[0], dtype=self.dtype,
                    kernel_regularizer='l2', use_bias=False,
                    input_shape = self.input_shape),
                  B(),
                  A("relu")]

        # Hidden Layers
        for u in units[1:-1]:
            layers.extend([D(u, dtype=self.dtype,
                             kernel_regularizer='l2', use_bias=False),
                           B(),
                           A("relu")])

        # Output Layer
        layers.append(D(units[-1], dtype=self.dtype))

        self.network = S(layers)

        self.train_data = None
        self.batch_size = None

        if noise_variance < 0.0:
            raise ValueError("`noise_variance` must be positive.")
        self.noise_variance = noise_variance * tf.eye(self.network.layers[-1].units)


    def compile(self,*args,**kwargs):
        """
        Compile network

        Any arguments to be passed to tf.keras.Model.compile
        """
        return self.network.compile(*args,**kwargs)

    def fit(self,x,y,batch_size: int=32,*args,**kwargs):
        """
        Fit network

        Parameters
        ----------
        x : array-like
            Network input values
        y : array-like
            Network output values
        batch_size : int, optional
            Mini batch size. The default size is 32.
        *args
            Arguments to be passed to tf.keras.Model.fit
        **kwargs
            Keyword arguments to be passed to tf.keras.Model.fit

        Raises
        ------
        ValueError
            If batch_size is different from that of the previous fit.
        """
        x = self._ensure_input(x)
        y = self._ensure_output(y)
        if self.train_data is None:
            self.train_data = tf.data.Dataset.from_tensor_slices(x)
        else:
            self.train_data.concatenate(tf.data.Dataset.from_tensor_slices(x))

        if self.batch_size and self.batch_size != batch_size:
            raise ValueError("Batch Size is inconsistent with the previous training")

        self.batch_size = batch_size
        kwargs["batch_size"] = batch_size
        return self.network.fit(x,y,*args,**kwargs)

    def predict(self, x, n_batches: int=100):
        """
        Predict Mean and Covariance

        Parameters
        ----------
        x : array-like
            Input Values
        n_batches : int, optional
            Number of batches to calculate mean and variance. The default is 100

        Returns
        -------
        m : tf.Tensor
            Mean of prediction. [batch size, output units]
        cov : tf.Tensor
            Covariance of prediction. [batch size, output units, output units]
        """
        x = self._ensure_input(x)
        batch_size = tf.constant(self.batch_size,dtype=tf.int64)
        n_batches = tf.constant(n_batches,dtype=tf.int64)

        return self._predict(x,batch_size,n_batches)


    @tf.function
    def _predict(self, x: tf.Tensor, batch_size: tf.Tensor, n_batches: tf.Tensor):
        sum  = 0.0
        sum2 = 0.0

        for B in self._mini_bathes(batch_size, n_batches):
            x_ = tf.identity(x)
            for L in self.network.layers:
                if not isinstance(L, self.BN_class):
                    x_ = L(x_,training=False)
                    B  = L(B,training=False)
                else:
                    mL = tf.reduce_mean(B, axis=0, keepdims=True)
                    vL = (tf.reduce_mean(tf.square(B), axis=0, keepdims=True)
                          - tf.square(mL))

                    norm = tf.sqrt(1.0/(vL + 1e-12))

                    x_ = (x_ - mL) * norm
                    B  = (B  - mL) * norm

                    if L.scale:
                        x_ = L.gamma * x_
                        B  = L.gamma * B

                    if L.center:
                        x_ += L.beta
                        B  += L.beta

            sum  = sum  + x_
            sum2 = sum2 + tf.expand_dims(x_,1) * tf.expand_dims(x_,2)

        inv_batch = 1.0 / tf.cast(n_batches,sum.dtype)
        m = sum  * inv_batch
        v = sum2 * inv_batch - tf.expand_dims(m,1) * tf.expand_dims(m,2)

        return m, v + tf.expand_dims(self.noise_variance,0)


    @tf.function
    def _mini_bathes(self, batch_size: tf.Tensor, n_batches: tf.Tensor):
        n = self.train_data.cardinality()

        if n < 0: # -1: INFINITE_CARDINALITY, -2: UNKNOWN_CARDINALITY
            n = 1024

        _dataset = self.train_data.shuffle(len(self.train_data),
                                           reshuffle_each_iteration=True)
        _dataset = _dataset.repeat().batch(batch_size, drop_remainder=True)

        return _dataset.take(n_batches)
