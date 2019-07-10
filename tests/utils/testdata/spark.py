import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors as MLVectors


class PySparkTestData:

    def __init__(self, spark=None):
        if spark is None:

            spark = SparkSession.builder.appName("test").master("local").getOrCreate()

            # spark.conf.set("spark.executor.memory", "120g")
            spark.conf.set("spark.driver.memory", "120g")
            spark.conf.set("spark.task.cpu", "1")
            spark.conf.set("spark.executor.extraJavaOptions", "-XX:+UseCompressedOops")
            spark.conf.set("spark.python.worker.memory", "2g")

            self.spark = spark

        else:
            self.spark = spark

    def get_weight_data(self):
        """

        :return:
        """
        num_samples = 100

        df1 = pd.DataFrame(data=np.arange(num_samples), columns=["id1"])
        df1["v1"] = [MLVectors.dense([float(val) for val in np.random.random(5, ).tolist()]) for _ in range(num_samples)]
        df1["pred"] = df1["v1"].apply(lambda x: x.sum())

        sdf1 = self.spark.createDataFrame(df1)

        sdf2 = sdf1.withColumnRenamed("id1", "id2").withColumnRenamed("v1", "v2").drop("pred")

        return sdf1, sdf2

    def get_model_data(self):
        """

        :return:
        """
        df1 = pd.DataFrame(data=np.arange(100), columns=["id1"])

        for i in range(1, 5):
            df1["col_{}".format(i)] = [float(val) for val in np.random.random(100, ).tolist()]

        df1["pred"] = df1[["col_{}".format(i) for i in range(1, 5)]].sum(axis=1)

        sdf1 = self.spark.createDataFrame(df1)

        sdf2 = sdf1.withColumnRenamed("id1", "id2").withColumnRenamed("v1", "v2").drop("pred")

        return sdf1, sdf2

    def get_nn_data(self):
        """

        :return:
        """

        df1 = pd.DataFrame(data=np.arange(5), columns=["id1"])
        df1["v1"] = [MLVectors.dense(np.random.random(5,).tolist()) for _ in range(5)]

        sdf1 = self.spark.createDataFrame(df1)

        sdf2 = sdf1.withColumnRenamed("id1", "id2").withColumnRenamed("v1", "v2")

        return sdf1, sdf2


