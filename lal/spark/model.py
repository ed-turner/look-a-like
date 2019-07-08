# python standard library
from abc import ABCMeta, abstractmethod
from functools import reduce

# pyspark package
from pyspark.ml.feature import VectorAssembler, ElementwiseProduct, StandardScaler
from pyspark.ml.pipeline import Pipeline

# in-house packages
from .utils.asserts import AssertArgumentSparkDataFrame
from .nn import KNNPowerMatcher
from lal.utils.logger import LALLogger


class _LALModelBase(metaclass=ABCMeta):

    lal_logger = LALLogger(__name__)
    assertor = AssertArgumentSparkDataFrame()

    def __init__(self, k, p, input_cols=None, label_cols=None):
        """

        :param input_cols:
        :param label_cols:
        """

        if isinstance(p, float):
            if isinstance(k, int):
                self.matcher = KNNPowerMatcher(k, p)
            else:
                raise ValueError("This type matcher is not supported")
        else:
            raise ValueError("This distance is not supported")

        self.model = None
        self.weighter = None

        self.input_cols = input_cols
        self.label_cols = label_cols

    @lal_logger.log_error
    @assertor.assert_arguments
    def _get_matches(self, sdf1, sdf2):
        """

        :param sdf1:
        :param sdf2:
        :return:
        """

        matches_sdf = self.matcher.match(sdf1, sdf2)

        return matches_sdf

    @lal_logger.log_error
    @assertor.assert_arguments
    def fit(self, sdf):
        """

        :param sdf:
        :return:
        """

        if self.weighter is None:
            raise NotImplementedError("The weighter parameter has not been defined.")

        weights_arr = self.weighter.get_feature_importances(sdf)

        pipeline_lst = [VectorAssembler(inputCols=self.input_cols, outputCol="vec"),
                        StandardScaler(inputCol="vec", outputCol="standard_vec"),
                        ElementwiseProduct(scalingVec=weights_arr,
                                           inputCol='standard_vec', outputCol='scaled_vec')]

        _model = Pipeline(stages=pipeline_lst)
        model = _model.fit(sdf)

        self.model = model

        return self

    @abstractmethod
    def predict(self, sdf1, sdf2):
        """

        :param sdf1:
        :param sdf2:
        :return:
        """
        pass


class LALGBSparkRegressor(_LALModelBase):

    @_LALModelBase.lal_logger.log_error
    @_LALModelBase.assertor.assert_arguments
    def predict(self, sdf1, sdf2):
        """

        :param sdf1:
        :param sdf2:
        :return:
        """

        model = self.model
        label_cols = self.label_cols

        if isinstance(label_cols, str):
            label_cols = [label_cols]

        if model is None:
            raise NotImplementedError("The model was not fitted yet!")

        transformed_sdf1 = model.transform(sdf1).select(["id", "scaled_vec"]).withColumnRenamed("id", "id1")\
            .withColumnRenamed("scaled_vec", "v1")

        transformed_sdf2 = model.transform(sdf2).select(["id", "scaled_vec"]).withColumnRenamed("id", "id2") \
            .withColumnRenamed("scaled_vec", "v2")

        matches_sdf = self._get_matches(transformed_sdf1, transformed_sdf2)

        preds_sdf = matches_sdf.join(sdf1.select(["id"] + label_cols).withColumnRenamed("id", "id1"), "id1")\
            .groupby("id2").avg(label_cols).withColumnRenamed("id2", "id")

        for col in label_cols:
            preds_sdf = preds_sdf.withColumnRenamed("avg({})".format(col), col)

        return preds_sdf
