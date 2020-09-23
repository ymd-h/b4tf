import tensorflow as tf
import tensorflow_probability as tfp


__all__ = ["MCBN", "MCBNLayer"]

class MCBNLayer(tf.keras.layers.BatchNormalization):
    """
    Monte Carlo Batch Normalization Layer
    """
    pass


class MCBN:
    """
    Monte Carlo Batch Normalization

    References
    ----------
    M. Teye et al.,
    "Bayesian Uncertainty Estimation for Batch Normalized Deep Networks",
    aeXiv 1802.06455, 2018
    """
    def fit(self,x,y):
        if self.train_data is None:
            self.train_data = tf.data.Dataset.from_tensor_slices(x)
        else:
            self.train_data.concatenate(tf.data.Dataset.from_tensor_slices(x))

        return self.network.fit(x,y)

    def predict(self,x,batch_size: int=32,n_iterations: int=100):
        """
        Predict Mean and Variance

        Parameters
        ----------
        x : array-like
            Input Values
        batch_size : int, optional
            Mini batch size. The default value is 32
        n_iterations : int, optional
            Number of iterations to calculate mean and variance. The default is 100

        Returns
        -------
        m : tf.Tensor
            Mean of prediction
        v : tf.Tensor
            Variance of prediction
        """
        x = tf.constant(x)
        batch_size = tf.constant(batch_size)
        n_iterations = tf.constant(n_iterations)

        return self._fit(x,batch_size,n_iterations)
