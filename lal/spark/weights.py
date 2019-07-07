from abc import ABCMeta, abstractmethod, ABC

from .utils.asserts import AssertArgumentSparkDataFrame

from pyspark.ml.regression import GBTRegressor
from pyspark.ml.classification import GBTClassifier


class _LGBMWeightsBase(metaclass=ABCMeta):

    assertor = AssertArgumentSparkDataFrame()

    space_grid = None

    model = None

    @assertor.assert_arguments
    def _opt_params(self, sdf):
        pass


class GBMWeightRegressor(_LGBMWeightsBase):

    model = GBTRegressor()


class GBMWeightClassifier(_LGBMWeightsBase):

    model = GBTClassifier()

