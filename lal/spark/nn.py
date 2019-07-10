from abc import ABCMeta, abstractmethod, ABC

import numpy as np

import warnings

from .utils.asserts import AssertArgumentSparkDataFrame

import pyspark.sql.functions as F
from pyspark.sql import Row
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType
from pyspark.ml.linalg import VectorUDT
from pyspark.mllib.linalg import DenseMatrix, Vectors
from pyspark.mllib.linalg.distributed import IndexedRowMatrix, IndexedRow, RowMatrix


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


class _PowerDistance(_DistanceBase):
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

        @F.udf(DoubleType(), VectorUDT())
        def tmp(vec):
            return float((vec[0] - vec[1]).norm(p))

        return tmp

    @_DistanceBase.assertor.assert_arguments
    def calculate_distance(self, sdf1, sdf2):
        """
        This will calculate the distance between the vector-type columns of two spark dataframes

        :param sdf1: This is to have a columns id1 (dtype int) and v1 (dtype Vector)
        :param sdf2: This is to have a columns id2 (dtype int) and v2 (dtype Vector)
        :return:
        """

        # this is the cartesian join
        all_sdf = sdf1.crossJoin(sdf2)

        subtract_vector_udf = self.calculate_distance_udf()

        # this sums all the p-powered differences between the vectors, and then takes the 1/p power of the final sum
        dist_sdf = all_sdf.select("*", subtract_vector_udf(F.array('v1', 'v2')).alias('diff'))

        dist_sdf.persist()

        return dist_sdf


class _MahalanobisDistance(_DistanceBase):
    """

    """

    def __init__(self):
        _DistanceBase.__init__(self)
        warnings.warn("Currently, the distance measure does not match with what is provided in scipy.. However, this" +
                      " measure follows the definition of the Mahalanobis distance. This implies scipy and spark " +
                      "computes the covariance differently..")

    @_DistanceBase.assertor.assert_arguments
    def calculate_distance(self, sdf1, sdf2):
        """
        This will calculate the distance between the vector-type columns of two spark dataframes

        :param sdf1: This is to have a columns id1 (dtype int) and v1 (dtype Vector)
        :param sdf2: This is to have a columns id2 (dtype int) and v2 (dtype Vector)
        :return:
        """

        cov = RowMatrix(sdf1.select(["v1"]).withColumnRenamed("v1", "v")
                        .union(sdf2.select(["v2"]).withColumnRenamed("v2", "v")).rdd
                        .map(lambda row: Vectors.fromML(row.asDict()["v"]))).computeCovariance().toArray()

        x, v = np.linalg.eigh(cov)

        indices = 1e-10 <= x

        # we are trying to enfore the data types to be only python types
        n = int(v.shape[0])
        m = int(indices.sum())

        v_vals = [float(val) for val in v[:, indices].reshape(-1,).tolist()]

        v_spark = DenseMatrix(n, m, v_vals)

        x_vals = [float(val) for val in np.diag(x[indices] ** -0.5).reshape(-1,).tolist()]

        x_spark = DenseMatrix(m, m, x_vals)

        # we get the index to maintain the order
        _sdf1 = sdf1.rdd.zipWithIndex()\
            .map(lambda val_key: Row(id1=val_key[0].id1, v1=val_key[0].v1, index=val_key[1])).toDF()

        _sdf1.persist()

        _sdf2 = sdf2.rdd.zipWithIndex()\
            .map(lambda val_key: Row(id2=val_key[0].id2, v2=val_key[0].v2, index=val_key[1])).toDF()

        _sdf2.persist()

        # we get our indexed row matrix
        _sdf1_mat = IndexedRowMatrix(_sdf1.rdd.map(lambda row: IndexedRow(index=row.asDict()["index"],
                                                                          vector=Vectors.fromML(row.asDict()["v1"]))))

        _sdf2_mat = IndexedRowMatrix(_sdf2.rdd.map(lambda row: IndexedRow(index=row.asDict()["index"],
                                                                          vector=Vectors.fromML(row.asDict()["v2"]))))

        # we apply our transformation and then set it as our new variable
        _sdf1 = _sdf1.drop("v1").join(_sdf1_mat.multiply(v_spark).multiply(x_spark).rows\
                                      .map(lambda indexed_row: Row(index=indexed_row.index,
                                                                   v1=indexed_row.vector)).toDF(), "index")

        _sdf2 = _sdf2.drop("v2").join(_sdf2_mat.multiply(v_spark).multiply(x_spark).rows\
                                      .map(lambda indexed_row: Row(index=indexed_row.index,
                                                                   v2=indexed_row.vector)).toDF(), "index")

        @F.udf(DoubleType(), VectorUDT())
        def tmp(vec):
            return float(vec[0].squared_distance(vec[1])) ** 0.5

        all_sdf = _sdf1.crossJoin(_sdf2)

        dist_sdf = all_sdf.select("*", tmp(F.array('v1', 'v2')).alias('diff'))

        dist_sdf.persist()

        return dist_sdf


class _CosineDistance(_DistanceBase):
    """

    """

    def __init__(self):
        _DistanceBase.__init__(self)

    @_DistanceBase.assertor.assert_arguments
    def calculate_distance(self, sdf1, sdf2):
        """
        This will calculate the distance between the vector-type columns of two spark dataframes

        :param sdf1: This is to have a columns id1 (dtype int) and v1 (dtype Vector)
        :param sdf2: This is to have a columns id2 (dtype int) and v2 (dtype Vector)
        :return:
        """

        @F.udf(DoubleType(), VectorUDT())
        def tmp(vec):
            return 1.0 - (float(vec[0].dot(vec[1])) / float(vec[0].norm(2.) * vec[1].norm(2.)))

        all_sdf = sdf1.crossJoin(sdf2)

        dist_sdf = all_sdf.select("*", tmp(F.array('v1', 'v2')).alias('diff'))

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

        # this partitions the data by the id2 values and than sort each row by the difference
        window = Window.partitionBy(dist_sdf['id2']).orderBy(dist_sdf['diff'].asc())

        top_k_match_sdf = dist_sdf.select('*', F.rank().over(window).alias('rank'))\
            .filter("rank <= {}".format(k)).select(["id1", "id2"])

        top_k_match_sdf.persist()

        return top_k_match_sdf


class KNNPowerMatcher(_PowerDistance, _KNNMatcherBase):
    """
    This is the K-Nearest Neighbor algorithm with the p-norm distance measure.
    """

    def __init__(self, p, k):
        """

        :param p:
        :param k:
        """

        _KNNMatcherBase.__init__(self, k)
        _PowerDistance.__init__(self, p)


class KNNMahalanobisMatcher(_MahalanobisDistance, _KNNMatcherBase):
    """
    This is the K-Nearest Neighbor algorithm with the mahalanobis distance measure.
    """

    def __init__(self, k):
        """

        :param k:
        """

        _KNNMatcherBase.__init__(self, k)
        _MahalanobisDistance.__init__(self)


class KNNCosineMatcher(_CosineDistance, _KNNMatcherBase):
    """
    This is the K-Nearest Neighbor algorithm with the cosine distance measure.
    """

    def __init__(self, k):
        """

        :param k:
        """

        _KNNMatcherBase.__init__(self, k)
        _CosineDistance.__init__(self)

