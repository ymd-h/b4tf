import tensorflow as tf
import tensorflow_probability as tfp


__all__ = ["MCBN"]


class MCBN:
    """
    Monte Carlo Batch Normalization

    References
    ----------
    M. Teye et al.,
    "Bayesian Uncertainty Estimation for Batch Normalized Deep Networks",
    aeXiv 1802.06455, 2018
    """
    BN_class = tf.keras.layers.BatchNormalization

    def __init__(self,network: tf.keras.Model, noise_variance: float):
        """
        Initialize MCBN

        Parameters
        ----------
        network : tf.keras.Model
            Network using batch normalization
        noise_variance : float
            Variance of observation noise. (Hyper parameter)

        Raises
        ------
        ValueError
            If network has no `tf.keras.layers.BatchNormalization`
        """
        self.network = network
        self.noise_variance = tf.constant(noise_variance)
        self.train_data = None

        has_BN = False
        for l in self.network.layers:
            if isinstance(l,BN_class):
                has_BN = True
                break

        if not has_BN:
            raise ValueError(f"`network` must has "
                             "`tf.keras.layers.BatchNormalization`")


    def fit(self,x,y,*args,**kwargs):
        if self.train_data is None:
            self.train_data = tf.data.Dataset.from_tensor_slices(x)
        else:
            self.train_data.concatenate(tf.data.Dataset.from_tensor_slices(x))

        return self.network.fit(x,y,*args,**kwargs)

    def predict(self, x, batch_size: int=32, n_batches: int=100):
        """
        Predict Mean and Covariance

        Parameters
        ----------
        x : array-like
            Input Values
        batch_size : int, optional
            Mini batch size. The default value is 32
        n_batches : int, optional
            Number of batches to calculate mean and variance. The default is 100

        Returns
        -------
        m : tf.Tensor
            Mean of prediction
        cov : tf.Tensor
            Covariance of prediction
        """
        x = tf.constant(x)
        batch_size = tf.constant(batch_size)
        n_batches = tf.constant(n_batches)

        return self._predict(x,batch_size,n_batches)


    @tf.function
    def _predict(self, x: tf.Tensor, batch_size: tf.Tensor, n_batches: tf.Tensor):
        sum  = tf.constant(0.0)
        sum2 = tf.constant(0.0)
        for B in self._mini_bathes(batch_size, n_batches):
            for L in self.network.layers:
                if not isinstance(L, BN_class):
                    x = L(x)
                    B = L(B)
                else:
                    mL = tf.reduce_mean(B, axis=0, keepdims=True)
                    vL = (tf.reduce_mean(tf.square(B), axis=0, keepdims=True)
                          - tf.square(mL))

                    norm = tf.sqrt(1.0/(vL + 1e-12))

                    x = (x - mL) * norm
                    B = (B - mL) * norm

                    if L.scale:
                        x = self.gamma * x
                        B = self.gamma * B

                    if L.center:
                        x += self.beta
                        B += self.beta

            sum  += x
            sum2 += tf.square(x)

        m = sum / n_batches
        v = (sum2 / n_batches) - tf.square(m)

        return m, v


    @tf.function
    def _mini_bathes(self, batch_size: tf.Tensor, n_batches: tf.Tensor):
        n = self.train_data.cardinality()

        if n < 0: # -1: INFINITE_CARDINALITY, -2: UNKNOWN_CARDINALITY
            n = 1024

        _dataset = self.train_data.shuffle(len(self.train_data),
                                           reshuffle_each_iteration=True)
        _dataset = _dataset.repeat().batch(batch_size, drop_remainder=True)

        return _dataset.take(n_batches)
