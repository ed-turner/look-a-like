import unittest
import logging

import numpy as np

from tests.utils.testdata.spark import PySparkTestData
from tests.utils.testframework.spark import PySparkTestFramework

from lal.spark.nn import KNNPowerMatcher, KNNMahalanobisMatcher
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

        self.logger.info("The results: \n{}".format(top_matches_sdf.toPandas().to_string()))

        self.assert_all_close_spark_dataframe(top_matches_sdf.select(["id1"]), top_matches_sdf.select(["id2"]),
                                              ["id1"], ["id2"])

    pass


class TestKNNMahalanobisMatcher(PySparkTestFramework):

    logger = LALLogger("test_knn_mahalanobis_matcher")

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

        matcher = KNNMahalanobisMatcher(k)

        top_matches_sdf = matcher.match(sdf1, sdf2)

        self.logger.info("The results: \n{}".format(top_matches_sdf.toPandas().to_string()))

        self.assert_all_close_spark_dataframe(top_matches_sdf.select(["id1"]), top_matches_sdf.select(["id2"]),
                                              ["id1"], ["id2"])

    pass


if __name__ == "__main__":

    unittest.main()
