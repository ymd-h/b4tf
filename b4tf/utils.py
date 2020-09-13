from typing import Iterable

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
        if dtype:
            self.Gamma.dtype = dtype
        return 1.0/self.Gamma.sample(shape)
