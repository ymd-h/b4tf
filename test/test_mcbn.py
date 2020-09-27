import unittest

import tensorflow as tf
import tensorflow_probability as tfp

from b4tf.models import MCBN


class TestMCBN(unittest.TestCase):
    def test_raise_init(self):
        """
        Raise ValueError with negative noise
        """
        with self.assertRaises(ValueError):
            MCBN([5,1], -0.05)


    def test_fit_predict(self):
        mcbn = MCBN([5,1], 0.5,input_shape=(2,))
        mcbn.compile("adam","mean_squared_error")

        x = tf.constant([2.0,1.0])
        y = tf.constant(1.0)
        x_ = tf.constant([2.0,1.0])

        mcbn.fit(x,y)

        m, cov = mcbn.predict(x_)
        self.assertIsInstance(m,tf.Tensor)
        self.assertEqual(m.shape,(1,1))
        self.assertIsInstance(cov,tf.Tensor)
        self.assertEqual(cov.shape,(1,1,1))

        m, cov = mcbn.predict(x_,n_batches=2)
        self.assertIsInstance(m,tf.Tensor)
        self.assertEqual(m.shape,(1,1))
        self.assertIsInstance(cov,tf.Tensor)
        self.assertEqual(cov.shape,(1,1,1))

    def test_deep_fit_predict(self):
        mcbn = MCBN([5,4,5,1], 0.5,input_shape=(2,))
        mcbn.compile("adam","mean_squared_error")

        x = tf.constant([2.0,1.0])
        y = tf.constant(1.0)
        x_ = tf.constant([2.0,1.0])

        mcbn.fit(x,y)

        m, cov = mcbn.predict(x_)
        self.assertIsInstance(m,tf.Tensor)
        self.assertEqual(m.shape,(1,1))
        self.assertIsInstance(cov,tf.Tensor)
        self.assertEqual(cov.shape,(1,1,1))

        m, cov = mcbn.predict(x_,n_batches=2)
        self.assertIsInstance(m,tf.Tensor)
        self.assertEqual(m.shape,(1,1))
        self.assertIsInstance(cov,tf.Tensor)
        self.assertEqual(cov.shape,(1,1,1))


    def test_fit_batch(self):
        mcbn = MCBN([5,3], 0.5,input_shape=(2,))
        mcbn.compile("adam","mean_squared_error")

        x = tf.constant([[2.0,1.0],
                         [2.0,1.5],
                         [1.1,1.2]])
        y = tf.constant([[1.0,1.0,1.0],
                         [2.0,2.0,2.0],
                         [3.0,3.0,3.0]])

        mcbn.fit(x,y)


    def test_predict_batch(self):
        mcbn = MCBN([5,3], 0.5,input_shape=(2,))
        mcbn.compile("adam","mean_squared_error")

        x = tf.constant([[2.0,1.0],
                         [2.0,1.5],
                         [1.1,1.2]])
        y = tf.constant([[1.0,1.0,1.0],
                         [2.0,2.0,2.0],
                         [3.0,3.0,3.0]])
        x_ = tf.constant([[2.0,1.0],
                          [1.0,1.0],
                          [2.2,2.2],
                          [3.0,1.0]])

        mcbn.fit(x,y)

        m, cov = mcbn.predict(x_)
        self.assertIsInstance(m,tf.Tensor)
        self.assertEqual(m.shape,(4,3))
        self.assertIsInstance(cov,tf.Tensor)
        self.assertEqual(cov.shape,(4,3,3))

        m, cov = mcbn.predict(x_,n_batches=10)
        self.assertIsInstance(m,tf.Tensor)
        self.assertEqual(m.shape,(4,3))
        self.assertIsInstance(cov,tf.Tensor)
        self.assertEqual(cov.shape,(4,3,3))


if __name__ == "__main__":
    unittest.main()
