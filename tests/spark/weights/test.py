import unittest

from tests.utils.testframework.spark import PySparkTestFramework

from lal.spark.weights import GBMWeightClassifier, GBMWeightRegressor


class TestWeighter(PySparkTestFramework):

    pass


if __name__ == "__main__":

    unittest.main()
