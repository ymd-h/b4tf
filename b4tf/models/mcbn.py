import tensorflow as tf
import tensorflow_probability as tfp


__all__ = ["MCBN", "MCBNLayer"]

class MCBNLayer(tf.keras.layers.BatchNormalization):
    """
    Monte Carlo Batch Normalization Layer
    """



class MCBN:
    """
    Monte Carlo Batch Normalization

    References
    ----------
    M. Teye et al.,
    "Bayesian Uncertainty Estimation for Batch Normalized Deep Networks",
    aeXiv 1802.06455, 2018
    """
