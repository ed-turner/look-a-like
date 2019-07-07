from abc import ABCMeta, abstractmethod, ABC

from .utils.asserts import AssertArgumentSparkDataFrame

import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.linalg import Vector, DenseMatrix, VectorUDT


class _DistanceBase(metaclass=ABCMeta):
    """
    This is a base class to help organize the code-base
    """

    assertor = AssertArgumentSparkDataFrame()

    @abstractmethod
    def calculate_distance(self, sdf1, sdf2):
        """
        This will calculate the distance between the vector-type columns of two spark dataframes

        :param sdf1:
        :param sdf2:
        :return:
        """
        pass


class PowerDistance(_DistanceBase):
    """

    """

    def __init__(self, p):
        """

        :param p:
        """
        self.p = p

    def calculate_distance_udf(self):
        """
        This will return a decorated function that is a udf, to help calculate the distance between two vectors. This
        is element wise subtraction with each difference brought to the power of p.
        :return:
        """

        p = self.p

        @F.udf(VectorUDT())
        def tmp(vec):
            return (vec[0] - vec[1]) ** p

        return tmp

    @_DistanceBase.assertor.assert_arguments
    def calculate_distance(self, sdf1, sdf2):
        """
        This will calculate the distance between the vector-type columns of two spark dataframes

        :param sdf1: This is to have a columns id1 (dtype int) and v1 (dtype Vector)
        :param sdf2: This is to have a columns id2 (dtype int) and v2 (dtype Vector)
        :return:
        """

        p = self.p

        # this is the cartesian join
        all_sdf = sdf1.crossJoin(sdf2)

        subtract_vector_udf = self.calculate_distance_udf()

        dist_sdf = all_sdf.select("*", (F.sum(*subtract_vector_udf(F.array('v1', 'v2'))) ** (1.0 / p)).alias('diff'))

        dist_sdf.persist()

        return dist_sdf


class _KNNMatcherBase(_DistanceBase, ABC):

    def __init__(self, k):
        self.k = k

    def match(self, sdf1, sdf2):
        """
        This will match the two spark dataframes together
        :param sdf1: The training dataset
        :type sdf1: pyspark.sql.dataframe.DataFrame
        :param sdf2: The testing dataset
        :type sdf2: pyspark.sql.dataframe.DataFrame
        :return top_k_match_sdf: The top k matches for the testing dataset.
        :rtype top_k_match_sdf: pyspark.sql.dataframe.DataFrame
        """

        k = self.k

        dist_sdf = self.calculate_distance(sdf1, sdf2)

        top_k_match_sdf = dist_sdf.groupby("id2").agg(F.sort_array(F.col("diff")).limit(k)).select(["id1", "id2"])

        top_k_match_sdf.persist()

        return top_k_match_sdf


class KNNPowerMatcher(PowerDistance, _KNNMatcherBase):
    """

    """

    def __init__(self, p, k):
        """

        :param p:
        :param k:
        """

        _KNNMatcherBase.__init__(self, k)
        PowerDistance.__init__(self, p)

