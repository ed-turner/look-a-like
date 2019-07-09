import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors as MLVectors


class PySparkTestData:

    def __init__(self, spark=None):
        if spark is None:
            self.spark = SparkSession.builder.appName("test").master("local[*]").getOrCreate()

        else:
            self.spark = spark

    @classmethod
    def get_weight_data(cls):
        pass

    @classmethod
    def get_match_data(cls):
        pass

    def get_nn_data(self):
        """

        :return:
        """

        df1 = pd.DataFrame(data=np.arange(5), columns=["id1"])
        df1["v1"] = [MLVectors.dense(np.random.random(5,).tolist()) for _ in range(5)]

        sdf1 = self.spark.createDataFrame(df1)

        sdf2 = sdf1.withColumnRenamed("id1", "id2").withColumnRenamed("v1", "v2")

        return sdf1, sdf2


