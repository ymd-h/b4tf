import unittest

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization

from b4tf.models import MCBN


class TestMCBN(unittest.TestCase):
    def test_raise_init(self):
        """
        Raise ValueError without having tf.keras.layers.BatchNormalization
        """

        with self.assertRaises(ValueError):
            MCBN(Sequential([Dense(5,input_shape=(2,))]), 0.05)


    def test_fit_predict(self):
        mcbn = MCBN(Sequential([Dense(5,input_shape=(2,),activation="relu"),
                                BatchNormalization(),
                                Dense(1)]),
                    0.5)

        x = tf.constant([2.0,1.0])
        y = tf.constant(1.0)
        x_ = tf.constant((2.0,1.0))

        mcbn.fit(x,y)
        mcbn.predict(x_)

if __name__ == "__main__":
    unittest.main()
