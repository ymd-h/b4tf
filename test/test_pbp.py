import unittest

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

        self.assertTrue(tf.reduce_all(pbp._logZ(diff2,v) == -0.5*tf.math.log(2.0*pi*v)))


    def test_fit(self):
        pbp = PBP([2,2,1],input_shape=(2,))

        x = tf.constant([1.0,2.0])
        y = tf.constant(0.5)
        pbp.fit(x,y)


        x2 = tf.constant([[1.0,2.0],
                          [2.0,3.0]])
        y2 = tf.constant([0.5,0.2])
        pbp.fit(x2,y2)


    def test_call(self):
        pbp = PBP([2,2,1],input_shape=(3,))

        x1 = tf.constant([1.0,2.0,3.0])
        pbp(x1)

        x2 = tf.constant([[1.0,2.0,3.0],
                          [2.0,3.0,4.0]])
        pbp(x2)


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


    def test_sample_weight(self):
        layer = self._class(5)
        layer.build(3)
        w, b = layer._sample_weights()

        self.assertEqual(w.shape,(3,5))
        self.assertEqual(b.shape,(5,))

    def test_predict(self):
        layer = self._class(5)
        layer.build(3)

        m_prev = tf.constant([[1.0,2.0,3.0],
                              [2.0,3.0,4.0]])
        v_prev = tf.constant([[1.0,2.0,3.0],
                              [2.0,3.0,4.0]])
        m, v = layer.predict(m_prev,v_prev)

    def test_predict_without_variance(self):
        layer = self._class(5)
        layer.build(3)

        m_prev = tf.constant([[1.0,2.0,3.0],
                              [2.0,3.0,4.0]])
        v_prev = tf.zeros_like(m_prev)
        m, v = layer.predict(m_prev,v_prev)


class TestPBPReLULayer(unittest.TestCase):
    _class = PBPReLULayer
    def test_relu_call(self):
        layer = self._class(5)
        layer.build(3)

        y = layer(tf.constant([[1.0,2.0,3.0],
                               [2.0,3.0,4.0]]))
        self.assertTrue(tf.reduce_all(y>=0.0))

if __name__ == "__main__":
    unittest.main()
