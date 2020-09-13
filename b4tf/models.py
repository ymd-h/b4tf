from typing import Iterable

import tensorflow as tf
from tensorflow.python.framework import tensor_shape

import tensorflow_probability as tfp


class PBPLayer(tf.keras.layers.Layer):
    """
    Layer for Probabilistic Backpropagation
    """
    def __init__(self,units: int,
                 *args,**kwargs):
        """
        Initialize PBP layer

        Parameters
        ----------
        units: int
           Number of units in layer. (Output shape)
        """
        super().__init__(*args,**kwargs)
        self.units = units
        pi = tf.math.atan(tf.constant(1.0)) * 4
        self.log_inv_sqrt2pi = -0.5*tf.math.log(2.0*pi)

    def build(self,input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `PBPLayer` '
                             'should be defined. Found `None`.')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        self.inv_sqrtV1 = 1.0 / tf.math.sqrt(last_dim + 1)
        self.inv_V1 = tf.math.square(self.inv_sqrtV1)


        self.kernel_m = self.add_weight("kernel_mean",
                                        shape=[last_dim,self.units],
                                        initializer=tf.keras.initializers.HeNormal(),
                                        dtype=self.dtype,
                                        trainable=True)
        self.kernel_v = self.add_weight("kernel_variance",
                                        shape=[last_dim,self.units],
                                        initializer=,
                                        dtype=self.dtype,
                                        trainable=True)
        self.bias_m = self.add_weight("bias_mean",
                                      shape=[self.units,],
                                      initializer=tf.keras.initializers.HeNormal(),
                                      dtype=self.dtype,
                                      trainable=True)
        self.bias_v = self.add_weight("bias_variance",
                                      shape=[self.units,],
                                      initializer=,
                                      dtype=self.dtype,
                                      trainable=True)
        self.Normal = tfp.distribution.Normal(loc=0.0, scale=1.0)
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
        self.kernel_v.assign_sub(tf.math.square(self.kernel_v) *
                                 (tf.math.square(dlogZ_dkm) - 2*dlogZ_dkv))

        # Bias
        self.bias_m.assign_add(self.bias_v * dlogZ_dbm)
        self.bias_v.assign_sub(tf.math.squre(self.bias_v) *
                               (tf.math.square(dlogZ_dbm) - 2*dlogZ_dbv))

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
            Input

        Returns
        -------
        z : tf.Tensor
            Output
        """
        W, b = self._sample_weights()
        return (tf.tensordot(W,x,axes=[-1,1]) + b)*self.inv_sqrtV1

    def get_config(self):
        return {**super().get_config(),}


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
        m : tf.Tensor
            Mean
        v : tf.Tensor
            Variance
        """
        m = ((tf.tensordot(self.kernel_m,m_prev,axes=[-1,0]) + self.bias_m)
             * self.inv_sqrtV1)

        v = ((tf.tensordot(tf.math.square(self.kernel_m),v_prev,axes=[-1,0]) +
              tf.tensordot(self.kernel_v,tf.math.square(m_prev),axes=[-1,0]) +
              self.bias_v +
              tf.tensordot(self.kernel_v,v_prev,axes=[-1,0]))
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
            Input

        Returns
        -------
        z : tf.Tensor
            Output
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
        ma, va = seuper().predict(m_prev,v_prev)

        _sqrt_v = tf.math.sqrt(va)
        _alpha = ma / _sqrt_v
        _inv_alpha = 1.0/_alpha
        _cdf_alpha = self.Normal.cdf(_alpha)
        _gamma = tf.where(_alpha < -30,
                          -_alpha + _inv_alpha * (-1 + 2*tf.math.squre(_inv_alpha)),
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
        self.alpha_y  = tf.Variable(1.0,trainable=True)
        self.beta_y   = tf.Variable(0.0,trainable=True)


        self.input_shape = input_shape
        self.call_shape = (-1,*self.input_shape)

        last_shape = self.input_shape
        self.layers = []
        for u in units[:-1]:
            # Hidden Layer's Activation is ReLU
            l = PBPReLULayer(u)
            l.build(last_shape)
            self.layers.append(l)
            last_shape = u
        else:
            # Output Layer's Activation is Linear
            l = PBPLayer(units[-1])
            l.build(last_shape)
            self.layers.append(l)


        self.Normal = tfp.distributions.Norma(loc=0.0,scale=1.0)
        self.Gamma = tfp.distributions.Gamma(concentration=self.alpha_y,
                                             rate=self.beta_y)

    def _logZ(self,diff_square: tf.Tensor, v: tf.Tensor):
        return tf.reduce_sum(-0.5 * (diff_square / v) +
                             self.log_inv_sqrt2pi - tf.matn.log(v))


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
        x = tf.convert_to_tensor(x,shape=self.call_shape)

        trainables = [l.trainable_weights for l in self.layers]
        with tf.GradientTape() as tape:
            tape.watch(trainables)
            m, v = self._predict(x)

            v0 = v + self.beta_y/(self.alpha_y - 1)
            diff_square = tf.math.square(y - m)
            logZ0 = self._logZ(diff_square,v0)

        grad = tape.gradient(logZ0,trainables)
        for l, g in zip(self.layers, grad):
            l.apply_gradient(g)


        alpha1 = self.alpha_y + 1
        v1 = v + self.beta_y/self.alpha_y
        v2 = v + self.beta_y/alpha1

        logZ1 = self._logZ(diff_square,v1)
        logZ2 = self._logZ(diff_square,v2)

        logZ2_logZ1 = logZ2 - logZ1
        logZ1_logZ0 = logZ1 - logZ0
        # Must update beta first
        self.beta_y.assign(self.beta_y/(tf.math.exp(logZ2_logZ1)*alpha1 -
                                        tf.math.exp(logZ1_logZ0)*self.alpha_y))
        self.alpha_y.assign(1.0/(tf.math.exp(logZ2_logZ1 - logZ1_logZ0) *
                                 alpha1/self.alpha_y  - 1.0))

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
        x = tf.convert_to_tensor(x,shape=self.call_shape)
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
        x = tf.convert_to_tensor(x,shape=self.call_shape)
        m, v = self._predict(x)

        # TODO:
        # Convolute by Normal(0,Gamma(alpha,beta)^{-1})

        return m, v

    @tf.function
    def _predict(self,x: tf.Tensor):
        m, v = x, tf.zeros_like(x)
        for l in self.layers:
            m, v = l.predict(m,v)

        return m, v
