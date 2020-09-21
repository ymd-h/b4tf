import unittest

class TestB4TF(unittest.TestCase):
    def test_import_b4tf(self):
        import b4tf as _b4tf
        pbp = _b4tf.models.PBP([2,1])

    def test_import_PBP(self):
        from b4tf.models import PBP as _PBP
        pbp = _PBP([2,1])

    def test_import_models(self):
        import b4tf.models as _models
        pbp = _models.PBP([2,1])

if __name__ == "__main__":
    unittest.main()
