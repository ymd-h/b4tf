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
        self.Normal = tfp.distribution.Normal(loc=0.0, scale=1.0)
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

    @tf.function
    def logZ0_logZ1_logZ2(self):
        """
        Calculate LogZ

        Returns
        -------
        logZ0 : tf.Tensor
            Log Z(alpha,beta)
        LogZ1 : tf.Tensor
            Log Z(alpha+1,beta)
        LogZ2 : tf.Tensor
            Log Z(alpha+2,beta)
        """
        zeros_m = tf.zeros_like(self.kernel_m)
        zeros_b = tf.zeros_like(self.bias_m)
        alpha1 = self.alpha + 1
        alpha2 = self.alpha + 2
        return (self._logZ(self.kernel_m,self.alpha,self.beta,zeros_m,self.kernel_v) +
                self._logZ(self.bias_m,self.alpha,self.beta,zeros_b,self.bias_v),
                self._logZ(self.kernel_m,alpha1,self.beta,zeros_m,self.kernel_v) +
                self._logZ(self.bias_m,alpha1,self.beta,zeros_b,self.bias_v),
                self._logZ(self.kernel_m,alpha2,self.beta,zeros_m,self.kernel_v) +
                self._logZ(self.bias_m,alpha2,self.beta,zeros_b,self.bias_v))

    @tf.function
    def update_weight(self):
        # Kernel
        with tf.GradientTape() as g:
            g.watch([self.kernel_m,self.kernel_v])
            logZ = self._logZ(self.kernel_m,
                              self.alpha,self.beta,
                              tf.zeros_like(self.kernel_m),self.kernel_v)
        dlogZ_dm, dlogZ_dv = g.gradient(logZ,[self.kernel_m,self.kernel_v])
        self.kernel_m.assign_add(self.kernel_v * dlogZ_dm)
        self.kernel_v.assign_sub(tf.math.square(self.kernel_v) *
                                 (tf.math.square(dlogZ_dm) - 2*dlogZ_dv))

        # Bias
        with tf.GradientTape() as g:
            g.watch([self.bias_m,self.bias_v])
            logZ = self._logZ(self.bias_m,
                              self.alpha,self.beta,
                              tf.zeros_like(self.bias_m),self.bias_v)
        dlogZ_dm, dlogZ_dv = g.gradient(logZ,[self.bias_m,self.bias_v])
        self.bias_m.assign_add(self.bias_v * dlogZ_dm)
        self.bias_v.assign_sub(tf.math.squre(self.bias_v) *
                               (tf.math.square(dlogZ_dm) - 2*dlogZ_dv))

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
        W, b = self._sample_weights()
        return (tf.tensordot(W,x,axes=[-1,1]) + b)*self.inv_sqrtV1

    def fit(self,x,y):
        logZ0 = tf.constant(0.0)
        logZ1 = tf.constant(0.0)
        logZ2 = tf.constant(0.0)
        for l in self.layers:
            l.update_weight()
            _logZ0, _logZ1, _logZ2 = l.logZ0_logZ1_logZ2()
            logZ0 += _logZ0
            logZ1 += _logZ1
            logZ2 += _logZ2


        alpha1 = self.alpha + 1
        inv_beta = 1.0/self.beta_lambda
        alpha_inv_beta = self.alpha * inv_beta
        alpha1_inv_beta = alpha_inv_beta + inv_beta
        logZ2_logZ1 = logZ2 - logZ1
        logZ1_logZ0 = logZ1 - logZ0
        # Must update beta first
        self.beta_lambda.assign(1.0/(tf.math.exp(logZ2_logZ1)*alpha1_inv_beta -
                                     tf.math.exp(logZ1_logZ0)*alpha_inv_beta))
        self.alpha_lambda.assign(1.0/(tf.math.exp(logZ2_logZ1 - logZ1_logZ0) *
                                      alpha1/self.alpha_lambda  - 1.0))


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
        self.beta_gamma   = tf.Variable(0.0,trainable=True)

        self.alpha_lambda = tf.Variable(1.0,trainable=True)
        self.beta_lambda  = tf.Variable(0.0,trainable=True)


        self.input_shape = input_shape

        last_shape = self.input_shape
        self.layers = []
        for u in units:
            l = PBPLayer(u,self.alpha_lambda,self.beta_lambda)
            l.build(last_shape)
            self.layers.append(l)
            last_shape = u

    def _logZ(self,y: tf.Tensor,
              alpha: tf.Tensor,beta: tf.Tensor,
              m: tf.Tensor,v: tf.Tensor):
        pass

    def fit(self,x,y):
        pass

    def __call__(self,x):
        x = tf.convert_to_tensor(x,shape=(-1,*self.input_shape))
        return self._call(x)

    @tf.function
    def _call(self,x:tf.Tensor):
        for l in self.layers[:-1]:
            x = l(x)
            x = tf.maximum(x,tf.zeros_like(x)) # ReLU

        return self.layers[-1](x)
