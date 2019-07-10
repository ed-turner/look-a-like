import unittest

import numpy as np
from pyspark.sql import SparkSession


class PySparkTestFramework(unittest.TestCase):

    spark = SparkSession.builder.appName("test").master("local[*]").getOrCreate()
    spark.conf.set("spark.executor.memory", "120g")
    spark.conf.set("spark.driver.memory", "120g")
    spark.conf.set("spark.task.cpu", "1")
    spark.conf.set("spark.executor.extraJavaOptions", "-XX:+UseCompressedOops")
    spark.conf.set("spark.python.worker.memory", "2g")
    spark.conf.set("spark.executor.cores", "1")

    def assert_all_close_spark_dataframe(self, sdf1, sdf2, cols1, cols2):
        """

        :param sdf1:
        :param sdf2:
        :return:
        """

        assert np.allclose(sdf1.toPandas()[cols1].astype(float), sdf2.toPandas()[cols2].astype(float))

