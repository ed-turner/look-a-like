from abc import ABCMeta, abstractmethod
from pyspark.ml.linalg import Vector, DenseMatrix


class DistanceBase(metaclass=ABCMeta):
    """
    This is a base class to help organize the code-base
    """

    @abstractmethod
    def calculate_distance(self, df1, df2):
        """

        :param df1:
        :param df2:
        :return:
        """