import unittest

import numpy as np
from scipy.spatial.distance import cdist

from tests.utils.testdata.spark import PySparkTestData
from tests.utils.testframework.spark import PySparkTestFramework

from lal.spark.nn import KNNPowerMatcher, KNNMahalanobisMatcher, KNNCosineMatcher
from lal.utils.logger import LALLogger


class TestKNNPowerMatcher(PySparkTestFramework):

    logger = LALLogger("test_knn_power_matcher")

    @logger.log_error
    def test_1nn(self):
        """

        :return:
        """

        self.logger.info("We are testing that the data is mapped to itself.")

        p = np.random.random()*100

        spark = self.spark

        if spark is None:
            raise NotImplementedError("The spark session is none!")

        generator = PySparkTestData(spark=spark)

        sdf1, sdf2 = generator.get_nn_data()

        k = 1

        matcher = KNNPowerMatcher(p, k)

        top_matches_sdf = matcher.match(sdf1, sdf2)

        self.assert_all_close_spark_dataframe(top_matches_sdf.select(["id1"]), top_matches_sdf.select(["id2"]),
                                              ["id1"], ["id2"])

    @logger.log_error
    def test_distance(self):
        self.logger.info("We are comparing the distance calculation to what is available in scipy")

        spark = self.spark

        if spark is None:
            raise NotImplementedError("The spark session is none!")

        p = np.random.random() * 100

        generator = PySparkTestData(spark=spark)

        sdf1, sdf2 = generator.get_nn_data()

        k = 1

        arr1 = np.vstack((val.toArray() for val in sdf1.orderBy("id1").select(["v1"]).toPandas()["v1"].tolist()))

        matcher = KNNPowerMatcher(p, k)

        dist_sdf = matcher.calculate_distance(sdf1, sdf2)

        dists = dist_sdf.orderBy("id1").groupby("id1").pivot("id2").max("diff").fillna(0.0).drop("id1").toPandas().values

        scipy_dist = cdist(arr1, arr1, 'minkowski', p=p)

        assert np.allclose(dists, scipy_dist)


class TestKNNMahalanobisMatcher(PySparkTestFramework):

    logger = LALLogger("test_knn_mahalanobis_matcher")

    @logger.log_error
    def test_1nn(self):
        """

        :return:
        """

        self.logger.info("We are testing that the data is mapped to itself.")

        spark = self.spark

        if spark is None:
            raise NotImplementedError("The spark session is none!")

        generator = PySparkTestData(spark=spark)

        sdf1, sdf2 = generator.get_nn_data()

        k = 1

        matcher = KNNMahalanobisMatcher(k)

        top_matches_sdf = matcher.match(sdf1, sdf2)

        self.assert_all_close_spark_dataframe(top_matches_sdf.select(["id1"]), top_matches_sdf.select(["id2"]),
                                              ["id1"], ["id2"])

    @logger.log_error
    def test_distance(self):
        self.logger.info("We are comparing the distance calculation to what is available in scipy")

        spark = self.spark

        if spark is None:
            raise NotImplementedError("The spark session is none!")

        generator = PySparkTestData(spark=spark)

        sdf1, sdf2 = generator.get_nn_data()

        k = 1

        arr1 = np.vstack((val.toArray() for val in sdf1.orderBy("id1").select(["v1"]).toPandas()["v1"].tolist()))

        matcher = KNNMahalanobisMatcher(k)

        dist_sdf = matcher.calculate_distance(sdf1, sdf2)

        dists = dist_sdf.orderBy("id1").groupby("id1").pivot("id2").max("diff").fillna(0.0).drop("id1").toPandas().values

        scipy_dist = cdist(arr1, arr1, 'mahalanobis')

        assert np.allclose(dists, scipy_dist)


class TestKNNCosineMatcher(PySparkTestFramework):

    logger = LALLogger("test_knn_cosine_matcher")

    @logger.log_error
    def test_1nn(self):
        """

        :return:
        """

        self.logger.info("We are testing that the data is mapped to itself.")

        spark = self.spark

        if spark is None:
            raise NotImplementedError("The spark session is none!")

        generator = PySparkTestData(spark=spark)

        sdf1, sdf2 = generator.get_nn_data()

        k = 1

        matcher = KNNCosineMatcher(k)

        top_matches_sdf = matcher.match(sdf1, sdf2)

        self.assert_all_close_spark_dataframe(top_matches_sdf.select(["id1"]), top_matches_sdf.select(["id2"]),
                                              ["id1"], ["id2"])

    @logger.log_error
    def test_distance(self):
        self.logger.info("We are comparing the distance calculation to what is available in scipy")

        spark = self.spark

        if spark is None:
            raise NotImplementedError("The spark session is none!")

        generator = PySparkTestData(spark=spark)

        sdf1, sdf2 = generator.get_nn_data()

        k = 1

        arr1 = np.vstack((val.toArray() for val in sdf1.orderBy("id1").select(["v1"]).toPandas()["v1"].tolist()))

        matcher = KNNCosineMatcher(k)

        dist_sdf = matcher.calculate_distance(sdf1, sdf2)

        dists = dist_sdf.orderBy("id1").groupby("id1").pivot("id2").max("diff").fillna(0.0).drop("id1").toPandas().values

        scipy_dist = cdist(arr1, arr1, 'cosine')

        assert np.allclose(dists, scipy_dist)


if __name__ == "__main__":

    unittest.main()
