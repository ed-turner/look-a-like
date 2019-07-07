import unittest

from pyspark.sql import SparkSession


class PySparkTestFramework(unittest.TestCase):

    @classmethod
    def get_spark_session(cls):
        return SparkSession.builder.master().appName().getOrCreate()

    def assert_all_close_spark_dataframe(self, sdf1, sdf2):
        """

        :param sdf1:
        :param sdf2:
        :return:
        """
        pass

