import numpy as np

from .utils.asserts import AssertArgumentSparkDataFrame

from pyspark.ml.regression import GBTRegressor
from pyspark.ml.classification import GBTClassifier

from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


class _LGBMWeightsBase:

    assertor = AssertArgumentSparkDataFrame()

    evaluator = None

    model = None

    def __init__(self, feature_col, label_col):
        self.feature_col = feature_col
        self.label_col = label_col

    def get_space_grid(self):
        model = self.model

        if model is None:
            raise NotImplementedError("The model parameter is not set.")

        return ParamGridBuilder() \
            .addGrid(model.stepSize, (10.0 ** np.arange(-10.0, 1.0, 2.0)).tolist()) \
            .addGrid(model.subsamplingRate, (np.arange(2.0, 11.0, 2.0) / 10.0).tolist()) \
            .addGrid(model.maxDepth, [1, 3, 5, 7, 9, 11]) \
            .addGrid(model.maxIter, [100, 500, 1000, 5000, 10000])\
            .build()

    @assertor.assert_arguments
    def get_feature_importances(self, sdf):
        """

        :param sdf:
        :return:
        """

        evaluator = self.evaluator

        if evaluator is None:
            raise NotImplementedError("The evaluator parameter is not set.")

        space_grid = self.get_space_grid()

        model = self.model

        crossval = CrossValidator(estimator=model,
                                  estimatorParamMaps=space_grid,
                                  evaluator=evaluator,
                                  numFolds=5)

        cvModel = crossval.fit(sdf)

        return cvModel.bestModel.featureImportances


class GBMWeightRegressor(_LGBMWeightsBase):
    """
    This object will derive the feature importance weights based on a continuous output.  It will optimize the
    Gradient Boosting Model based on a regression metric, and then return the featureImportances based on the most
    optimized result.
    """
    def __init__(self, **kwargs):
        _LGBMWeightsBase.__init__(self, **kwargs)

        model = GBTRegressor(featuresCol=self.feature_col, labelCol=self.label_col)
        evaluator = RegressionEvaluator(labelCol=self.label_col, metricName='mse')

        self.model = model
        self.evaluator = evaluator


class GBMWeightBinaryClassifier(_LGBMWeightsBase):
    """
    This object will derive the feature importance weights based on binary output.  It will optimize the
    Gradient Boosting Model based on a classification metric, and then return the featureImportances based on the most
    optimized result.
    """
    def __init__(self, **kwargs):
        _LGBMWeightsBase.__init__(self, **kwargs)

        model = GBTClassifier(featuresCol=self.feature_col, labelCol=self.label_col)
        evaluator = BinaryClassificationEvaluator(labelCol=self.label_col, metricName='areaUnderROC')

        self.model = model
        self.evaluator = evaluator


class GBMWeightMultiClassifier(_LGBMWeightsBase):
    """
    This object will derive the feature importance weights based on a multiclass output.  It will optimize the
    Gradient Boosting Model based on a classification metric, and then return the featureImportances based on the most
    optimized result.
    """
    def __init__(self, **kwargs):
        _LGBMWeightsBase.__init__(self, **kwargs)

        model = GBTClassifier(featuresCol=self.feature_col, labelCol=self.label_col)
        evaluator = MulticlassClassificationEvaluator(labelCol=self.label_col, metricName='f1')

        self.model = model
        self.evaluator = evaluator

