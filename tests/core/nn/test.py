import unittest

import numpy as np
from scipy.spatial.distance import cdist
from lal.core.nn import KNNMahalanobisMatcher, KNNPowerMatcher, KNNCosineMatcher
from lal.utils.logger import LALLogger


class TestKNNPowerMatcher(unittest.TestCase):

    logger = LALLogger("test_knn_power_matcher")

    @logger.log_error
    def test_1nn(self):

        self.logger.info("We are comparing the expected matches to the resulting matches")

        arr1 = np.random.random((10, 20))

        matcher = KNNPowerMatcher(1, 100*np.random.random())

        matches = matcher.match(arr1, arr1)

        assert np.allclose(matches[:, 0], np.arange(10).reshape(-1, 1).astype(np.int64)[:, 0])

    @logger.log_error
    def test_distance(self):
        self.logger.info("We are comparing the distance calculation to what is available in scipy")
        arr1 = np.random.random((10, 20))

        p = max(100 * np.random.random(), 1)

        matcher = KNNPowerMatcher(1, p)

        dists = matcher.calc_dist(arr1, arr1)

        scipy_dist = cdist(arr1, arr1, 'minkowski', p=p)

        assert np.allclose(dists, scipy_dist)


class TestKNNMahalanobisMatcher(unittest.TestCase):

    logger = LALLogger("test_knn_mahalanobis_matcher")

    @logger.log_error
    def test_1nn(self):

        self.logger.info("We are comparing the expected matches to the resulting matches")

        arr1 = np.random.random((10, 20))

        matcher = KNNMahalanobisMatcher(1)

        matches = matcher.match(arr1, arr1)

        assert np.allclose(matches[:, 0], np.arange(10).reshape(-1, 1).astype(np.int64)[:, 0])

    @logger.log_error
    def test_distance(self):
        self.logger.info("We are comparing the distance calculation to what is available in scipy")
        arr1 = np.random.random((50, 20))

        matcher = KNNMahalanobisMatcher(1)

        dists = matcher.calc_dist(arr1, arr1)

        scipy_dist = cdist(arr1, arr1, 'mahalanobis')

        assert np.allclose(dists, scipy_dist)


class TestKNNCosineMatcher(unittest.TestCase):

    logger = LALLogger("test_knn_cosine_matcher")

    @logger.log_error
    def test_1nn(self):
        self.logger.info("We are comparing the expected matches to the resulting matches")

        arr1 = np.random.random((10, 20))

        matcher = KNNCosineMatcher(1)

        matches = matcher.match(arr1, arr1)

        assert np.allclose(matches[:, 0], np.arange(10).reshape(-1, 1).astype(np.int64)[:, 0])

    @logger.log_error
    def test_distance(self):
        self.logger.info("We are comparing the distance calculation to what is available in scipy")
        arr1 = np.random.random((50, 20))

        matcher = KNNCosineMatcher(1)

        dists = matcher.calc_dist(arr1, arr1)

        scipy_dist = cdist(arr1, arr1, 'cosine')

        assert np.allclose(dists, scipy_dist)


if __name__ == "__main__":

    unittest.main()
