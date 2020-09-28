from typing import Iterable

import tensorflow as tf
import tensorflow_probability as tfp

class ReciprocalGammaInitializer:
    """
    Weight initializers by 1/Gamma(alpha,beta)
    """
    def __init__(self,alpha,beta):
        """
        Initialize ReciprocalGammaInitializer

        Parameters
        ----------
        alpha : int, float or tf.Tensor
            Parameter for Gamma(alpha,beta)
        beta : int, float or tf.Tensor
            Parameter for Gamma(alpha,beta)
        """
        self.Gamma = tfp.distributions.Gamma(concentration=alpha,rate=beta)

    def __call__(self,shape: Iterable,dtype=None):
        """
        Returns a tensor object initialized as specified by the initializer.

        Parameters
        ----------
        shape : Iterable
            Shape for returned tensor
        dtype : dtype
            dtype for returned tensor


        Returns
        -------
        : tf.Tensor
            Initialized tensor
        """
        g = 1.0/self.Gamma.sample(shape)
        if dtype:
            g = tf.cast(g,dtype=dtype)

        return g

@tf.function
def safe_div(x: tf.Tensor,y: tf.Tensor, eps:tf.Tensor = tf.constant(1e-6)):
    """
    Non overflow division for positive tf.Tensor

    Parameters
    ----------
    x : tf.Tensor
        Numerator
    y : tf.Tensor
        Denominator
    eps : tf.Tensor, optional
        Small positive value. The default is 1e-6

    Returns
    -------
    sign(y) * x/(|y|+eps) : tf.Tensor
        Results

    Notes
    -----
    User must guaruantee `eps >= 0`
    """
    _eps = tf.cast(eps,dtype=y.dtype)
    return x/(tf.where(y >= 0, y + _eps, y - _eps))


@tf.function
def safe_exp(x: tf.Tensor, BIG: tf.Tensor = tf.constant(20)):
    """
    Non overflow exp(x)

    Parameters
    ----------
    x : tf.Tensor
        Input
    BIG : tf.Tensor, optional
        Maximum exponent. The default value is 20 (exp(x) <= 1e+20).

    Returns
    -------
    exp(min(x,BIG)) : tf.Tensor
        Results
    """
    return tf.math.exp(tf.math.minimum(x,tf.cast(BIG,dtype=x.dtype)))


@tf.function
def non_negative_constraint(x: tf.Tensor):
    return tf.maximum(x,tf.zeros_like(x))


def create_model(units,cls=tfp.layers.DenseReparameterization,
                 input_shape=(1,),activation=tf.nn.leaky_relu,**kwargs):
    """
    Create model (network) with specified size units

    Parameters
    ----------
    units : array-like of int
        Hidden and output units
    cls : tf.keras.Layer
        Class for each layer
    input_shape : array-like of int
        Input shape. The default value is `(1,)`
    activation : callable or str
        Activation function can be used at `tf.keras.Layer`

    Returns
    -------
    model : tf.keras.Model
        Model with specified units
    """
    S = tf.keras.Sequential
    return S([cls(units[0],activation=activation,input_shape=input_shape,**kwargs)] +
             [cls(i,activation=activation,**kwargs) for i in units[1:-1]] +
             [cls(units[-1],**kwargs)])


def multivariate_normal_fn(loc,relative=0.01):
    r"""
    Create a function for multivariate normal N(loc, abs(loc)*relative)

    This function can be used like `tfp.layers.util.default_multivariate_normal_fn`

    Parameters
    ----------
    loc : array-like
        Mean for normal function
    relative : float
        Standard deviation = abs(loc * relative)

    Returns
    -------
    _fn : callable
       Function creates Independent(Normal(loc,abs(loc*relative)))
    """
    def _fn(dtype, shape, name, trainable,add_variable_fn):
        del name, trainable, add_variable_fn   # unused
        w = tf.convert_to_tensor(loc, dtype=dtype)
        dist = tfp.distributions.Normal(loc=w, scale=tf.math.abs(w * relative))
        batch_ndims = tf.size(dist.batch_shape_tensor())
        return tfp.distributions.Independent(dist,reinterpreted_batch_ndims=batch_ndims)
    return _fn


def BNN_like(NN,cls=tfp.layers.DenseReparameterization,copy_weight=False,**kwargs):
    """
    Create Bayesian Neural Network like input Neural Network shape

    Parameters
    ----------
    NN : tf.keras.Model
        Neural Network for imitating shape
    cls : tfp.layers
        Bayes layers class
    copy_weight : bool, optional
        Copy weight from NN when `True`. The default is `False`


    Returns
    -------
    model : tf.keras.Model
        Bayes Neural Network
    """
    inputs = tf.keras.Input(shape=(tf.shape(NN.layers[0].kernel)[0],))
    x = inputs

    N = len(NN.layers)
    for i, L in enumerate(NN.layers):
        layer_kwargs = { **kwargs }

        if copy_weight:
            layer_kwargs["kernel_prior_fn": multivariate_normal_fn(L.kernel)]
            layer_kwargs["bias_prior_fn": multivariate_normal_fn(L.bias)]

        x = cls(L.units,activation=L.activation,**layer_kwargs)(x)

    return tf.keras.Model(inputs=inputs,outputs=x)
