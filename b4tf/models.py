from typing import Iterable, Union

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_shape

import tensorflow_probability as tfp

from b4tf.utils import ReciprocalGammaInitializer

class PBPLayer(tf.keras.layers.Layer):
    """
    Layer for Probabilistic Backpropagation
    """
    def __init__(self,units: int,
                 dtype=tf.float32,
                 *args,**kwargs):
        """
        Initialize PBP layer

        Parameters
        ----------
        units: int
           Number of units in layer. (Output shape)
        dtype : tf.dtypes.DType
           Data type
        """
        super().__init__(dtype=tf.as_dtype(dtype),*args,**kwargs)
        self.units = units

    def build(self,input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `PBPLayer` '
                             'should be defined. Found `None`.')
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: last_dim})
        self.inv_sqrtV1 = tf.cast(1.0 / tf.math.sqrt(1.0*last_dim + 1),
                                  dtype=self.dtype)
        self.inv_V1 = tf.math.square(self.inv_sqrtV1)


        over_gamma = ReciprocalGammaInitializer(1.0,1e-6)
        self.kernel_m = self.add_weight("kernel_mean",
                                        shape=[last_dim,self.units],
                                        initializer=tf.keras.initializers.HeNormal(),
                                        dtype=self.dtype,
                                        trainable=True)
        self.kernel_v = self.add_weight("kernel_variance",
                                        shape=[last_dim,self.units],
                                        initializer=over_gamma,
                                        dtype=self.dtype,
                                        trainable=True)
        self.bias_m = self.add_weight("bias_mean",
                                      shape=[self.units,],
                                      initializer=tf.keras.initializers.HeNormal(),
                                      dtype=self.dtype,
                                      trainable=True)
        self.bias_v = self.add_weight("bias_variance",
                                      shape=[self.units,],
                                      initializer=over_gamma,
                                      dtype=self.dtype,
                                      trainable=True)
        self.Normal=tfp.distributions.Normal(loc=tf.constant(0.0,dtype=self.dtype),
                                             scale=tf.constant(1.0,dtype=self.dtype))
        self.built = True

    @tf.function
    def apply_gradient(self,gradient):
        """
        Applay gradient and update weights and bias

        Parameters
        ----------
        gradient : list
            List of gradients for weights and bias.
            [d(logZ)/d(kernel_m), d(logZ)/d(kernel_v),
             d(logZ)/d(bias_m)  , d(logZ)/d(bias_v)]
        """

        dlogZ_dkm, dlogZ_dkv, dlogZ_dbm, dlogZ_dbv = gradient

        # Kernel
        self.kernel_m.assign_add(self.kernel_v * dlogZ_dkm)
        new_kv = (self.kernel_v -
                  (tf.math.square(self.kernel_v) *
                   (tf.math.square(dlogZ_dkm) - 2*dlogZ_dkv)))
        self.kernel_v.assign(tf.math.maximum(new_kv,tf.zeros_like(new_kv)))

        # Bias
        self.bias_m.assign_add(self.bias_v * dlogZ_dbm)
        new_bv = (self.bias_v -
                  (tf.math.square(self.bias_v) *
                   (tf.math.square(dlogZ_dbm) - 2*dlogZ_dbv)))
        self.bias_v.assign(tf.math.maximum(new_bv,tf.zeros_like(new_bv)))

    @tf.function
    def _sample_weights(self):
        eps = self.Normal.sample(self.kernel_m.shape)
        std = tf.math.sqrt(tf.math.maximum(self.kernel_v,
                                           tf.zeros_like(self.kernel_v)))
        W = self.kernel_m + std * eps

        eps = self.Normal.sample(self.bias_m.shape)
        std = tf.math.sqrt(tf.math.maximum(self.bias_v,
                                           tf.zeros_like(self.bias_v)))
        b = self.bias_m + std * eps
        return W, b

    @tf.function
    def call(self,x: tf.Tensor):
        """
        Calculate deterministic output

        Parameters
        ----------
        x : tf.Tensor
            Input. [batch, prev_units]

        Returns
        -------
        y : tf.Tensor
            Output. [batch, units]
        """
        W, b = self._sample_weights()
        return ((tf.tensordot(x,W,axes=[1,0]) + tf.expand_dims(b,axis=0))
                * self.inv_sqrtV1)

    def get_config(self):
        return {**super().get_config()}


    @tf.function
    def predict(self, m_prev: tf.Tensor, v_prev: tf.Tensor):
        """
        Predict Mean and Variance

        Parameters
        ----------
        m_prev : tf.Tensor
            Previous Mean. [batch, prev_units]
        v_prev : tf.Tensor
            Previous Variance. [batch, prev_units]

        Returns
        -------
        m : tf.Tensor
            Mean. [batch, units]
        v : tf.Tensor
            Variance. [batch, units]
        """
        m = ((tf.tensordot(m_prev,self.kernel_m,axes=[1,0]) +
              tf.expand_dims(self.bias_m,axis=0))
             * self.inv_sqrtV1)

        v = ((tf.tensordot(v_prev,tf.math.square(self.kernel_m),axes=[1,0]) +
              tf.tensordot(tf.math.square(m_prev),self.kernel_v,axes=[1,0]) +
              tf.expand_dims(self.bias_v,axis=0) +
              tf.tensordot(v_prev,self.kernel_v,axes=[1,0]))
             * self.inv_V1)

        return m, v


class PBPReLULayer(PBPLayer):
    @tf.function
    def call(self,x: tf.Tensor):
        """
        Calculate deterministic output

        Parameters
        ----------
        x : tf.Tensor
            Input. [batch, features]

        Returns
        -------
        z : tf.Tensor
            Output. [batch, features]
        """
        x = super().call(x)
        return tf.maximum(x,tf.zeros_like(x))

    @tf.function
    def predict(self, m_prev: tf.Tensor, v_prev: tf.Tensor):
        """
        Predict Mean and Variance

        Parameters
        ----------
        m_prev : tf.Tensor
            Previous Mean
        v_prev : tf.Tensor
            Previous Variance

        Returns
        -------
        mb : tf.Tensor
            Mean
        vb : tf.Tensor
            Variance
        """
        ma, va = super().predict(m_prev,v_prev)

        _sqrt_v = tf.math.sqrt(tf.math.maximum(va,tf.zeros_like(va)))
        _alpha = ma / _sqrt_v
        _inv_alpha = 1.0/_alpha
        _cdf_alpha = self.Normal.cdf(_alpha)
        _gamma = tf.where(_alpha < -30,
                          -_alpha + _inv_alpha * (-1 + 2*tf.math.square(_inv_alpha)),
                          self.Normal.prob(-_alpha)/_cdf_alpha)
        _vp = ma + _sqrt_v * _gamma

        mb = _cdf_alpha * _vp
        vb = (mb * _vp * self.Normal.cdf(-_alpha) +
              _cdf_alpha * va * (1 - _gamma * (_gamma + _alpha)))

        return mb, vb

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
                 input_shape: Iterable[int]=(1,),
                 dtype: Union[tf.dtypes.DType,np.dtype,str]=tf.float32):
        """
        Initialize PBP model

        Parameters
        ----------
        units : Iterable[int]
            Numbers of hidden units and outputs
        input_shape : Iterable[int], optional
            Input shape for PBP model. The default value is `(1,)`
        dtype : tf.dtypes.DType or np.dtype or str
            Data type
        """
        self.dtype = tf.as_dtype(dtype)
        self.alpha = tf.Variable(1.0,trainable=True,dtype=self.dtype)
        self.beta  = tf.Variable(0.0,trainable=True,dtype=self.dtype)

        pi = tf.math.atan(tf.constant(1.0,dtype=self.dtype)) * 4
        self.log_inv_sqrt2pi = -0.5*tf.math.log(2.0*pi)

        self.input_shape = tf.TensorShape(input_shape)
        self.call_rank = tf.rank(tf.constant(0,
                                             shape=self.input_shape,
                                             dtype=self.dtype)) + 1
        self.output_rank = units[-1] + 1

        last_shape = self.input_shape
        self.layers = []
        for u in units[:-1]:
            # Hidden Layer's Activation is ReLU
            l = PBPReLULayer(u,dtype=self.dtype)
            l.build(last_shape)
            self.layers.append(l)
            last_shape = u
        else:
            # Output Layer's Activation is Linear
            l = PBPLayer(units[-1],dtype=self.dtype)
            l.build(last_shape)
            self.layers.append(l)


        self.Normal=tfp.distributions.Normal(loc=tf.constant(0.0,dtype=self.dtype),
                                             scale=tf.constant(1.0,dtype=self.dtype))
        self.Gamma = tfp.distributions.Gamma(concentration=self.alpha,
                                             rate=self.beta)

    def _logZ(self,diff_square: tf.Tensor, v: tf.Tensor):
        return tf.reduce_sum(-0.5 * (diff_square / v) +
                             self.log_inv_sqrt2pi - tf.math.log(v))


    def _ensure_input(self,x):
        """
        Ensure input type and shape

        Parameters
        ----------
        input : Any
            Input values

        Returns
        -------
        x : tf.Tensor
            Input values with shape=(-1,*self.input_shape) and dtype=self.dtype
        """
        x = tf.constant(x,dtype=self.dtype)
        if tf.rank(x) < self.call_rank:
            x = tf.reshape(x,[-1,*self.input_shape.as_list()])
        return x

    def _ensure_output(self,y):
        """
        Ensure output type and shape

        Parameters
        ----------
        y : Any
            Output values

        Returns
        -------
        y : tf.Tensor
           Output values with shape=(-1,self.layers[-1].units) and dtype=self.dtype
        """
        y = tf.constant(y,dtype=self.dtype)
        if tf.rank(y) < self.output_rank:
            y = tf.reshape(y,[-1,self.layers[-1].units])
        return y


    def fit(self,x,y):
        """
        Fit posterior distribution with observation

        Parameters
        ----------
        x : array-like
            Observed input
        y : array-like
            Observed output
        """
        x = self._ensure_input(x)
        y = self._ensure_output(y)

        self._fit(x,y)

    @tf.function
    def _fit(self,x: tf.Tensor, y: tf.Tensor):
        trainables = [l.trainable_weights for l in self.layers]
        with tf.GradientTape() as tape:
            tape.watch(trainables)
            m, v = self._predict(x)

            v0 = v + self.beta/(self.alpha - 1)
            diff_square = tf.math.square(y - m)
            logZ0 = self._logZ(diff_square,v0)

        grad = tape.gradient(logZ0,trainables)
        for l, g in zip(self.layers, grad):
            l.apply_gradient(g)


        alpha1 = self.alpha + 1
        v1 = v + self.beta/self.alpha
        v2 = v + self.beta/alpha1

        logZ1 = self._logZ(diff_square,v1)
        logZ2 = self._logZ(diff_square,v2)

        logZ2_logZ1 = logZ2 - logZ1
        logZ1_logZ0 = logZ1 - logZ0
        # Must update beta first
        self.beta.assign(self.beta/(tf.math.exp(logZ2_logZ1)*alpha1 -
                                    tf.math.exp(logZ1_logZ0)*self.alpha))
        self.alpha.assign(1.0/(tf.math.exp(logZ2_logZ1 - logZ1_logZ0) *
                               alpha1/self.alpha  - 1.0))

    def __call__(self,x):
        """
        Calculate deterministic output

        Parameters
        ----------
        x : array-like
            Input

        Returns
        y : tf.Tensor
            Neural netork output
        """
        x = self._ensure_input(x)
        return self._call(x)

    @tf.function
    def _call(self,x:tf.Tensor):
        for l in self.layers:
            x = l(x)

        return x + (self.Normal.sample(x.shape) /
                    tf.math.sqrt(self.Gamma.sample(x.shape)))


    def predict(self,x):
        """
        Predict distribution

        Parameters
        ----------
        x : array-like
            Input

        Returns
        -------
        m : tf.Tensor
            Mean
        v : tf.Tensor
            Variance
        """
        x = self._ensure_input(x)
        m, v = self._predict(x)

        return m, v + self.beta/(self.alpha - 1)

    @tf.function
    def _predict(self,x: tf.Tensor):
        m, v = x, tf.zeros_like(x)
        for l in self.layers:
            m, v = l.predict(m,v)

        return m, v
