# python standard library
from abc import ABCMeta, abstractmethod
from functools import reduce

# pyspark package
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler, ElementwiseProduct, StandardScaler
from pyspark.ml.pipeline import Pipeline

# in-house packages
from .utils.asserts import AssertArgumentSparkDataFrame
from .weights import GBMWeightRegressor, GBMWeightBinaryClassifier, GBMWeightMultiClassifier
from .nn import KNNPowerMatcher, KNNCosineMatcher, KNNMahalanobisMatcher
from lal.utils.logger import LALLogger


class _LALModelBase(metaclass=ABCMeta):

    lal_logger = LALLogger(__name__)
    assertor = AssertArgumentSparkDataFrame()

    def __init__(self, k, p, input_cols=None, label_cols=None, optimize=True):
        """

        :param input_cols:
        :param label_cols:
        """

        if isinstance(p, float):
            assert 1.0 <= p

            if isinstance(k, int):
                self.matcher = KNNPowerMatcher(k, p)
            else:
                raise ValueError("This type matcher is not supported")
        elif p == 'cosine':
            if isinstance(k, int):
                self.matcher = KNNCosineMatcher(k)
            else:
                raise ValueError("This type matcher is not supported")

        elif p == 'mahalanbois':
            if isinstance(k, int):
                self.matcher = KNNMahalanobisMatcher(k)
            else:
                raise ValueError("This type matcher is not supported")
        else:
            raise ValueError("This distance is not supported")

        self.opt = optimize

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
    """
    This is when our training labels are continuous.
    """
    def __init__(self, **kwargs):
        _LALModelBase.__init__(self, **kwargs)
        self.weighter = GBMWeightRegressor(feature_col=self.input_cols, label_col=self.label_cols, optimize=self.opt)

    @_LALModelBase.lal_logger.log_error
    @_LALModelBase.assertor.assert_arguments
    def predict(self, sdf1, sdf2):
        """
        We predict the possible value our testing dataset will have, based on the continuous variables.

        :param sdf1: The training dataset
        :type sdf1: pyspark.sql.dataframe.DataFrame
        :param sdf2: The testing dataset
        :type sdf2: pyspark.sql.dataframe.DataFrame
        :return:
        """

        model = self.model
        label_cols = self.label_cols

        if isinstance(label_cols, str):
            label_cols = [label_cols]
        else:
            raise ValueError("At the moment, we only allow single output prediction.")

        if model is None:
            raise NotImplementedError("The model was not fitted yet!")

        transformed_sdf1 = model.transform(sdf1).select(["id", "scaled_vec"]).withColumnRenamed("id", "id1")\
            .withColumnRenamed("scaled_vec", "v1")

        transformed_sdf2 = model.transform(sdf2).select(["id", "scaled_vec"]).withColumnRenamed("id", "id2") \
            .withColumnRenamed("scaled_vec", "v2")

        matches_sdf = self._get_matches(transformed_sdf1, transformed_sdf2)

        exprs = {"{}".format(val): "avg" for val in label_cols}

        preds_sdf = matches_sdf.join(sdf1.select(["id"] + label_cols).withColumnRenamed("id", "id1"), "id1")\
            .groupby("id2").agg(exprs).withColumnRenamed("id2", "id")

        preds_sdf = preds_sdf.withColumnRenamed("avg({})".format(label_cols[0]), label_cols[0])

        return preds_sdf


class LALGBSparkBinaryClassifier(_LALModelBase):
    """
    This is when our training labels are binary.
    """
    def __init__(self, **kwargs):
        _LALModelBase.__init__(self, **kwargs)
        self.weighter = GBMWeightBinaryClassifier(feature_col=self.input_cols, label_col=self.label_cols, optimize=self.opt)

    @_LALModelBase.lal_logger.log_error
    @_LALModelBase.assertor.assert_arguments
    def predict_proba(self, sdf1, sdf2):
        """
        This predicts the probability of our test data having any of the available labels in the training dataset

        :param sdf1: The training dataset
        :type sdf1: pyspark.sql.dataframe.DataFrame
        :param sdf2: The testing dataset
        :type sdf2: pyspark.sql.dataframe.DataFrame
        :return:
        """

        model = self.model
        label_cols = self.label_cols

        if isinstance(label_cols, str):
            label_cols = [label_cols]
        else:
            raise ValueError("At the moment, we only allow single output prediction.")

        if model is None:
            raise NotImplementedError("The model was not fitted yet!")

        transformed_sdf1 = model.transform(sdf1).select(["id", "scaled_vec"]).withColumnRenamed("id", "id1") \
            .withColumnRenamed("scaled_vec", "v1")

        transformed_sdf2 = model.transform(sdf2).select(["id", "scaled_vec"]).withColumnRenamed("id", "id2") \
            .withColumnRenamed("scaled_vec", "v2")

        matches_sdf = self._get_matches(transformed_sdf1, transformed_sdf2)

        exprs = {"{}".format(val): "avg" for val in label_cols}

        preds_sdf = matches_sdf.join(sdf1.select(["id"] + label_cols).withColumnRenamed("id", "id1"), "id1") \
            .groupby("id2").agg(exprs).withColumnRenamed("id2", "id")

        preds_sdf = preds_sdf.withColumnRenamed("avg({})".format(label_cols[0]), "raw_{}".format(label_cols[0]))

        return preds_sdf

    @_LALModelBase.lal_logger.log_error
    @_LALModelBase.assertor.assert_arguments
    def predict(self, sdf1, sdf2):
        """
        We choose most probable label our samples in the testing dataset has.

        :param sdf1: The training dataset
        :type sdf1: pyspark.sql.dataframe.DataFrame
        :param sdf2: The testing dataset
        :type sdf2: pyspark.sql.dataframe.DataFrame
        :return:
        """

        label_col = self.label_cols

        preds_sdf = self.predict_proba(sdf1, sdf2)

        tmp_sdf = preds_sdf.withColumn(label_col, F.when(F.col("raw_{}".format(label_col)) < 0.5, 0).otherwise(1))

        return tmp_sdf


class LALGBSparkMultiClassifier(_LALModelBase):
    """
    This is when our training labels are categorical.
    """
    def __init__(self, **kwargs):
        _LALModelBase.__init__(self, **kwargs)
        self.weighter = GBMWeightMultiClassifier(feature_col=self.input_cols, label_col=self.label_cols, optimize=self.opt)
        self.pred_cols = None

    @_LALModelBase.lal_logger.log_error
    @_LALModelBase.assertor.assert_arguments
    def predict_proba(self, sdf1, sdf2):
        """
        This predicts the probability of our test data having any of the available labels in the training dataset

        :param sdf1: The training dataset
        :type sdf1: pyspark.sql.dataframe.DataFrame
        :param sdf2: The testing dataset
        :type sdf2: pyspark.sql.dataframe.DataFrame
        :return:
        """

        model = self.model
        label_cols = self.label_cols

        if isinstance(label_cols, str):
            label_cols = [label_cols]
        else:
            raise ValueError("At the moment, we only allow single output prediction.")

        if model is None:
            raise NotImplementedError("The model was not fitted yet!")

        transformed_sdf1 = model.transform(sdf1).select(["id", "scaled_vec"]).withColumnRenamed("id", "id1") \
            .withColumnRenamed("scaled_vec", "v1")

        transformed_sdf2 = model.transform(sdf2).select(["id", "scaled_vec"]).withColumnRenamed("id", "id2") \
            .withColumnRenamed("scaled_vec", "v2")

        # gets the matches
        matches_sdf = self._get_matches(transformed_sdf1, transformed_sdf2)

        # gets the class label per match
        matched_labels_sdf = matches_sdf.join(sdf1.select(["id"] + label_cols).withColumnRenamed("id", "id1"), "id1")

        # gets the unique values
        unique_vals = sdf1.agg(F.countDistinct(F.col(label_cols[0]))).collect()

        # one-hot-encodes our class labels
        one_hot_encoded_vals_sdf = reduce(lambda df, val:
                                          df.withColumn("{}_{}".format(label_cols[0],
                                                                       F.when(F.col(label_cols[0]) == val,
                                                                              1).otherwise(0))),
                                          unique_vals, matched_labels_sdf)

        # generates the experimental probability of belonging to that class
        exprs = {"{}_{}".format(label_cols[0], val): "avg" for val in unique_vals}

        tmp_preds_sdf = one_hot_encoded_vals_sdf.groupby("id2").agg(exprs).withColumnRenamed("id2", "id")

        # sums the independent-class probabilities to normalize the probabilities of the multiclass probs
        preds_sdf = reduce(lambda df, key: df.withColumnRenamed("avg({})".format(key), "raw_{}_tmp".format(key)),
                           list(exprs.keys()), tmp_preds_sdf).withColumn("one_norm", F.sum(F.col(key) for key in exprs.keys()))

        res_sdf = reduce(lambda df, key: df.withColumn("raw_{}".format(key),
                                                       F.col("raw_{}_tmp".format(key)) / F.col("one_norm")),
                         list(exprs.keys()), preds_sdf).drop("one_norm")

        self.pred_cols = list(exprs.keys())

        return res_sdf

    @_LALModelBase.lal_logger.log_error
    @_LALModelBase.assertor.assert_arguments
    def predict(self, sdf1, sdf2):
        """
        We choose most probable label our samples in the testing dataset has.

        :param sdf1: The training dataset
        :type sdf1: pyspark.sql.dataframe.DataFrame
        :param sdf2: The testing dataset
        :type sdf2: pyspark.sql.dataframe.DataFrame
        :return:
        """

        label_col = self.label_cols

        preds_cols = self.pred_cols

        preds_sdf = self.predict_proba(sdf1, sdf2)

        cond = "F.when" + ".when".join(
            ["(F.col('" + c + "') == F.col('max_value'), F.lit('" + c + "'))" for c in preds_cols])

        tmp_sdf = preds_sdf.withColumn("max_value", F.greatest(*(F.col(c) for c in preds_cols)))\
            .withColumn(label_col, eval(cond))

        return tmp_sdf
