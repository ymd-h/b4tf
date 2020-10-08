import tensorflow as tf
from tensorflow.python.framework import tensor_shape


__all__ = ["DenseNF"]

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

        self.built = True
