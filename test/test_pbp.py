import unittest

import numpy as np
import tensorflow as tf
from b4tf.models import PBP, PBPLayer, PBPReLULayer


class TestPBP(unittest.TestCase):
    def test_init(self):
        pbp = PBP([2,1])

        self.assertEqual(len(pbp.layers),2)
        self.assertIsInstance(pbp.layers[0],PBPReLULayer)
        self.assertIsInstance(pbp.layers[1],PBPLayer)
        self.assertEqual(pbp.input_shape,(1,))


    def test_logZ(self):
        pbp = PBP([2,1])

        diff2 = tf.constant(0.0)
        v = tf.constant(1.0)
        pi = tf.math.atan(1.0) * 4

        Z = pbp._logZ(diff2,v)
        e = -0.5*tf.math.log(2.0*pi*v)
        np.testing.assert_allclose(np.asarray(Z), np.asarray(e),rtol=1e-5)


    def test_fit(self):
        pbp = PBP([2,2,1],input_shape=(2,))

        x = tf.constant([1.0,2.0])
        y = tf.constant(0.5)
        pbp.fit(x,y)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(pbp.alpha)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(pbp.beta)))


        x2 = tf.constant([[1.0,2.0],
                          [2.0,3.0]])
        y2 = tf.constant([0.5,0.2])
        pbp.fit(x2,y2)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(pbp.alpha)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(pbp.beta)))


    def test_call(self):
        pbp = PBP([2,2,1],input_shape=(3,))

        x1 = tf.constant([1.0,2.0,3.0])
        y1 = pbp(x1)
        self.assertEqual(y1.shape,(1,1))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(y1)))

        m1,v1 = pbp.predict(x1)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(m1)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(v1)))
        self.assertTrue(tf.reduce_all(v1 >= 0))

        pbp.fit(x1,y1)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(pbp.alpha)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(pbp.beta)))

        x2 = tf.constant([[1.0,2.0,3.0],
                          [2.0,3.0,4.0]])
        y2 = pbp(x2)
        self.assertEqual(y2.shape,(2,1))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(y2)))

        m2, v2 = pbp.predict(x2)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(m2)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(v2)))
        self.assertTrue(tf.reduce_all(v2 >= 0))

        pbp.fit(x2,y2)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(pbp.alpha)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(pbp.beta)))

    def test_call_with_different_dtype(self):
        pbp = PBP([2,2,1],input_shape=(2,))

        x1 = np.arange(2)
        y1 = pbp(x1)
        self.assertEqual(y1.dtype,tf.float32)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(y1)))

    def test_dtype(self):
        pbp = PBP([2,1],dtype=tf.float64)

        x1 = np.asarray(1.0)
        y1 = pbp(x1)
        self.assertEqual(y1.dtype,tf.float64)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(y1)))

        m,v = pbp.predict(x1)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(m)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(v)))
        self.assertTrue(tf.reduce_all(v >= 0))

        pbp.fit(x1,y1)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(pbp.alpha)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(pbp.beta)))


class TestPBPLayer(unittest.TestCase):
    _class = PBPLayer
    def test_init(self):
        layer = self._class(10)
        self.assertEqual(layer.units,10)

    def test_build(self):
        layer = self._class(5)
        layer.build(3)
        self.assertTrue(layer.built)

        v = layer.trainable_weights

        self.assertTrue(tf.reduce_all(layer.kernel_m == v[0]))
        self.assertTrue(tf.reduce_all(layer.kernel_v == v[1]))
        self.assertTrue(tf.reduce_all(layer.bias_m   == v[2]))
        self.assertTrue(tf.reduce_all(layer.bias_v   == v[3]))


    def test_call(self):
        layer = self._class(5)
        layer.build(3)

        y = layer(tf.constant([[1.0,2.0,3.0],
                               [2.0,3.0,4.0]]))
        self.assertEqual(y.shape,(2,5))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(y)))


    def test_sample_weight(self):
        layer = self._class(5)
        layer.build(3)
        w, b = layer._sample_weights()

        self.assertEqual(w.shape,(3,5))
        self.assertEqual(b.shape,(5,))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(w)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(b)))

    def test_predict(self):
        layer = self._class(5)
        layer.build(3)

        m_prev = tf.constant([[1.0,2.0,3.0],
                              [2.0,3.0,4.0]])
        v_prev = tf.constant([[1.0,2.0,3.0],
                              [2.0,3.0,4.0]])
        m, v = layer.predict(m_prev,v_prev)
        self.assertEqual(m.shape,(2,5))
        self.assertEqual(v.shape,(2,5))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(m)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(v)))
        self.assertTrue(tf.reduce_all(v >= 0))

    def test_predict_without_variance(self):
        layer = self._class(5)
        layer.build(3)

        m_prev = tf.constant([[1.0,2.0,3.0],
                              [2.0,3.0,4.0]])
        v_prev = tf.zeros_like(m_prev)
        m, v = layer.predict(m_prev,v_prev)
        self.assertEqual(m.shape,(2,5))
        self.assertEqual(v.shape,(2,5))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(m)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(v)))
        self.assertTrue(tf.reduce_all(v >= 0))


class TestPBPReLULayer(unittest.TestCase):
    _class = PBPReLULayer
    def test_relu_call(self):
        layer = self._class(5)
        layer.build(3)

        y = layer(tf.constant([[1.0,2.0,3.0],
                               [2.0,3.0,4.0]]))
        self.assertTrue(tf.reduce_all(y>=0.0))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(y)))

if __name__ == "__main__":
    unittest.main()
