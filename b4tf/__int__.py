import tensorflow as tf
import tensorflow_probability as tfp


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
    """
    Create a function for multivariate normal N(loc, abs(loc)*relative)

    This function can be used like `tfp.layers.util.default_multivariate_normal_fn`

    Parameters
    ----------
    loc : array-like
        Mean for normal function
    relative : float
        Standard deviation = |loc * relative|

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


def BNN_like(NN,cls=DenseReparameterization,copy_weight=False,**kwargs):
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
