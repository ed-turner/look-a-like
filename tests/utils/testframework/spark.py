import unittest

import numpy as np
from pyspark.sql import SparkSession


class PySparkTestFramework(unittest.TestCase):

    spark = SparkSession.builder.appName("test").master("local[*]").getOrCreate()

    def assert_all_close_spark_dataframe(self, sdf1, sdf2, cols1, cols2):
        """

        :param sdf1:
        :param sdf2:
        :return:
        """

        assert np.allclose(sdf1.toPandas()[cols1].astype(float), sdf2.toPandas()[cols2].astype(float))

