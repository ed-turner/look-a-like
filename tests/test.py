import warnings
import unittest

try:
    from .spark.weights.test import TestWeighter
    from .spark.nn.test import TestKNNPowerMatcher
except ImportError:
    warnings.warn("The spark package was not installed in this installation.. If you want to have spark implementation"
                  ", please install the package lal[spark]")


if __name__ == "__main__":

    unittest.main()
