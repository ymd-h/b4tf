from typing import Iterable, Union

import tensorflow as tf
from tensorflow.python.framework import tensor_shape

from .base import ModelBase

__all__ = ["MaskedRealNVPFlow","DenseNF","MNF"]


class MaskedRealNVPFlow(ModelBase):
    """
    Masked Real NVP flow
    """
    def __init__(self,units :Iterable[int], *,
                 input_shape: Iterable[int]=(1,),
                 dtype: Union[tf.dtypes.DType,np.dtype,str]=tf.float32):
        """
        Initialize Masked Real NVP

        Parameters
        ----------
        units : Iterable[int]
            Numbers of hidden units and outputs
        input_shape : Iterable[int], optional
            Input shape for PBP model. The default value is `(1,)`
        dtype : tf.dtypes.DType or np.dtype or str
            Data type
        """
        super().__init__(dtype,input_shape,units[-1])
        self.bernoulli = tfp.distributions.Bernoulli(probs=0.5,dtype=self.dtype)

        last_shape = self.input_shape
        self._f = []
        for u in units:
            # Hidden Layer's Activation is tanh
            l = tf.keras.layers.Dense(u,dtype=self.dtype,activation="tanh")
            l.build(last_shape)
            self._f.append(l)
            last_shape = u

        self._g = tf.keras.layers.Dense(self.input_shape)
        self._g.build(u)

        self._k = tf.keras.layers.Dense(self.input_shape,activation="sigmoid")
        self._k.build(u)


    def __call__(self,z):
        """
        Calculate deterministic output

        Parameters
        ----------
        z : array-like
            Samples from distribution

        Returns
        -------
        z1 : tf.Tensor
            Samples from converted distribution
        LogDet : tf.Tensor
            Log determinant of Jaccobian
        """
        z = tf.constant(z,dtype=self.dtype)
        return self._call(z)


    @tf.function
    def _call(self, z: tf.Tensof):
        m = self.bernoulli.sample(shape=z.shape)
        h0 = m * z
        not_m = 1 - m

        h = h0
        for _f in self._f:
            h = _f(h) # tanh is included

        mu = self._g(h)
        sigma = self._k(h) # sigmoid is included

        z1 = h0 + not_m * (z * sigma + (1-sigma) * mu)
        LogDet = tf.reduce_sum(not_m * tf.math.log(sigma))
        return z1, LogDet


class DenseNF(tf.keras.layers.Layer):
    """
    Layer for Normalizing Flow
    """
    def __init__(self,units: int,
                 dtype=tf.float32,
                 *args,**kwargs):
        """
        Initialize NF layer

        Parameters
        ----------
        units : int
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
            raise ValueError('The last dimension of the inputs to `DnseNF` '
                             'should be defined. Found `None`.')
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: last_dim})

        over_gamma = ReciprocalGammaInitializer(6.0,6.0)
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
        self.built = True


class MNF(ModelBase):
    """
    Multiplicative Normalizing Flow

    References
    ----------
    C. Louis and M. Welling,
    "Multiplicative Normalizing Flows for Variational Bayesian Neural Networks",
    arXiv 1703.01961, 2017
    """
    def __init__(self,units: Iterable[int],*,
                 input_shape: Iterable[int]=(1,),
                 dtype: Union[tf.dtypes.DType,np.dtype,str]=tf.float32):
        """
        Initialize MNF model

        Parameters
        ----------
        units : Iterable[int]
            Numbers of hidden units and outputs
        input_shape : Iterable[int], optional
            Input shape for PBP model. The default value is `(1,)`
        dtype : tf.dtypes.DType or np.dtype or str
            Data type
        """
        super().__init__(dtype,input_shape,units[-1])

        last_shape = self.input_shape
        self.layers = []
        for u in units[:-1]:
            # Hidden Layer's Activation is ReLU
            l = DenseNF(u,dtype=self.dtype)
            l.build(last_shape)
            self.layers.append(l)
            self.layers.append(tf.keras.layers.Activation("relu"))
            last_shape = u

        # Output Layer's Activation is Linear
        l = DenseNF(units[-1],dtype=self.dtype)
        l.build(last_shape)
        self.layers.append(l)


    def fit(self,x,y,batch_size:int = 16):
        """
        Fit posterior distribution with observation

        Parameters
        ----------
        x : array-like
            Observed input
        y : array-like
            Observed output
        batch_size : int, optional
            Batch size. The default value is 16.
        """
        x = self._ensure_input(x)
        y = self._ensure_output(y)

        data = tf.data.Dataset.from_tensor_slices((x,y)).batch(batch_size)

        for _x,_y in data:
            self._fit(_x,_y)


    @tf.function
    def _fit(self,x: tf.Tensor, y: tf.Tensor):
        pass


    def __call__(self,x):
        """
        Calculate deterministic output

        Parameters
        ----------
        x : array-like
            Input

        Returns
        -------
        y : tf.Tensor
            Neural netork output
        """
        x = self._ensure_input(x)
        return self._call(x)

    @tf.function
    def _call(self,x: tf.Tensor):
        pass
