import unittest

class TestB4TF(unittest.TestCase):
    def test_import_b4tf(self):
        import b4tf as _b4tf
        pbp = _b4tf.models.PBP([2,1])
        mcbn = _b4tf.models.MCBN([2,1],0.5)

    def test_import_PBP(self):
        from b4tf.models import PBP as _PBP, MCBN as _MCBN
        pbp = _PBP([2,1])
        mcbn = _MCBN([2,1],0.5)

    def test_import_models(self):
        import b4tf.models as _models
        pbp = _models.PBP([2,1])
        mcbn = _models.MCBN([2,1],0.5)

if __name__ == "__main__":
    unittest.main()
