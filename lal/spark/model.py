# python standard library
from abc import ABCMeta, abstractmethod, ABC
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
    def get_matches(self, sdf1, sdf2):
        """

        :param sdf1:
        :param sdf2:
        :return:
        """
        model = self.model

        if model is None:
            raise NotImplementedError("The model was not fitted yet!")

        transformed_sdf1 = model.transform(sdf1).select(["id", "scaled_vec"]).withColumnRenamed("id", "id1") \
            .withColumnRenamed("scaled_vec", "v1")

        transformed_sdf2 = model.transform(sdf2).select(["id", "scaled_vec"]).withColumnRenamed("id", "id2") \
            .withColumnRenamed("scaled_vec", "v2")

        matches_sdf = self.matcher.match(transformed_sdf1, transformed_sdf2)

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

        label_cols = self.label_cols

        if isinstance(label_cols, str):
            label_cols = [label_cols]
        else:
            raise ValueError("At the moment, we only allow single output prediction.")

        matches_sdf = self.get_matches(sdf1, sdf2)

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

        label_cols = self.label_cols

        if isinstance(label_cols, str):
            label_cols = [label_cols]
        else:
            raise ValueError("At the moment, we only allow single output prediction.")

        matches_sdf = self.get_matches(sdf1, sdf2)

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


class LALGBSparkCategoricalClassifier(_LALModelBase):
    """

    """
    def __init__(self, **kwargs):
        _LALModelBase.__init__(self, **kwargs)
        self.weighter = GBMWeightMultiClassifier(feature_col=self.input_cols,
                                                 label_col=self.label_cols, optimize=self.opt)
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

        label_cols = self.label_cols

        assert isinstance(label_cols, str)

        # gets the matches
        matches_sdf = self.get_matches(sdf1, sdf2)

        def get_label_col_prediction(_col):
            matched_labels_sdf = matches_sdf.join(sdf1.select(["id", _col]).withColumnRenamed("id", "id1"),
                                                  "id1")

            # gets the unique values
            unique_vals = sdf1.agg(F.countDistinct(F.col(_col))).collect()

            # one-hot-encodes our class labels
            one_hot_encoded_vals_sdf = reduce(lambda df, val:
                                              df.withColumn("{}_{}".format(_col, val),
                                                                           F.when(F.col(_col) == val,
                                                                                  1).otherwise(0)),
                                              unique_vals, matched_labels_sdf)

            new_cols = ["{}_{}".format(_col, val) for val in unique_vals]

            tmp_preds_sdf = one_hot_encoded_vals_sdf.groupby("id2")\
                .agg(*[F.avg(_new_col).alias(_new_col) for _new_col in new_cols])\
                .withColumn("one_norm", F.sum(*[F.col(_new_col) for _new_col in new_cols]))

            res_sdf = reduce(lambda df, col: df.withColumn("raw_{}".format(col),
                                                           F.col(col) / F.col("one_norm")),
                             new_cols, tmp_preds_sdf).drop("one_norm")

            return new_cols, res_sdf

        pred_cols, _sdf = get_label_col_prediction(label_cols)

        self.pred_cols = pred_cols

        return _sdf

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

        label_cols = self.label_cols

        preds_cols = self.pred_cols

        preds_sdf = self.predict_proba(sdf1, sdf2)

        def get_predictions(label_col):
            pred_col = filter(lambda x: label_col in x, preds_cols)

            cond_lst = [F.when(F.col(_x) == F.col("max_value"), F.lit(_x.split("_{}_".format(label_col))[-1]))
                        for _x in pred_col]

            cond = reduce(lambda left, right: left.otherwise(right), cond_lst)

            tmp_sdf = preds_sdf.withColumn("max_value", F.greatest(*(F.col(c) for c in preds_cols)))\
                .withColumn(label_col, cond)

            return tmp_sdf.select(["id2", label_col])

        return get_predictions(label_cols)


class _LALSparkMultiBase(metaclass=ABCMeta):
    """
    This is a Base for the MultiOutput Model.  We will properly define the base below
    """
    task_base = _LALModelBase

    def __init__(self, **kwargs):

        self.pred_multi_dict = None

        if kwargs is None:
            raise ValueError
        else:
            if 'label_cols' in kwargs.keys():
                params = {key: kwargs[key] for key in kwargs.keys() if not ('label_cols' == key)}

                self.task_multi_dict = {label_col: self.task_base(label_cols=label_col, **params) for label_col in
                                   kwargs["label_cols"]}

    def fit(self, sdf):
        """
        In this fit method, we map each of the underlying single columnar methods onto each column we want to predict
        :param sdf:
        :return:
        """

        task_multi_dict = self.task_multi_dict

        self.pred_multi_dict = {key: task_multi_dict[key].fit(sdf) for key in task_multi_dict.keys()}

        return sdf

    def predict(self, sdf1, sdf2):
        pred_dict = self.pred_multi_dict

        def get_predictions(label_col):
            return pred_dict[label_col].predict(sdf1, sdf2)

        return reduce(lambda left, right: left.join(right, "id2"), map(get_predictions, list(pred_dict.keys())))


class LALGBSparkMultiBinaryClassifier(_LALSparkMultiBase):
    """
    This is our Multioutput Binary Classifier, where our training labels are all binary.
    """
    task_base = LALGBSparkBinaryClassifier

    def predict_proba(self, sdf1, sdf2):
        pred_dict = self.pred_multi_dict

        def get_predictions(label_col):
            return pred_dict[label_col].predict_proba(sdf1, sdf2)

        return reduce(lambda left, right: left.join(right, "id2"), map(get_predictions, list(pred_dict.keys())))


class LALGBSparkMultiCategoricalClassifier(_LALSparkMultiBase):
    """
    This is our Multioutput Categorical Classifier, where our training labels are all binary.
    """
    task_base = LALGBSparkCategoricalClassifier

    def predict_proba(self, sdf1, sdf2):
        pred_dict = self.pred_multi_dict

        def get_predictions(label_col):
            return pred_dict[label_col].predict_proba(sdf1, sdf2)

        return reduce(lambda left, right: left.join(right, "id2"), map(get_predictions, list(pred_dict.keys())))


class LALGBSparkMultiRegressorClassifier(_LALSparkMultiBase):
    """
    This is our Multioutput Regressor, where our training labels are all binary.
    """
    task_base = LALGBSparkRegressor
