from abc import abstractmethod, ABCMeta

import numpy as np

from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# this is for the utility of logging and type assertion
from lal.utils.logger import LALLogger
from lal.utils.asserts import AssertArgumentNDArray

from .nn import KNNPowerMatcher, KNNCosineMatcher, NNLinearSumCosineMatcher, NNLinearSumPowerMatcher
from .weights import LGBMClassifierWeight, LGBMRegressorWeight


class _Scaler(TransformerMixin):
    """
    This class will scale an array with a numpy array based into the __init__
    """
    def __init__(self, scale):
        self.scale = scale

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        scale = self.scale

        if len(scale.shape) == 1:
            return np.dot(X, np.diag(scale))
        elif self.scale.shape[1] == X.shape[1]:
            return np.dot(X, scale)
        else:
            raise ValueError("The shape of the scaling factors and the input data is not the same")


class LALGBBaseModel(metaclass=ABCMeta):
    """
    This is a base model to help generate the Look-A-Like fitting structure.

    Options for k is a non-negative integer, or linear_sum.
    Options for p is a nonzero float, or cosine.
    """

    lal_logger = LALLogger(__name__)
    assertor = AssertArgumentNDArray()

    lal_logger.__doc__ = "This is the Look-A-Like Class Level Logger, which helps log events at different levels to " \
                         "the console."
    assertor.__doc__ = "This is the Look-A-Like class level argument assertion"

    def __init__(self, k, p):

        if isinstance(p, float):
            if isinstance(k, int):
                self.matcher = KNNPowerMatcher(k, p)
            elif k == "linear_sum":
                self.matcher = NNLinearSumPowerMatcher(p)
            else:
                raise ValueError("This type matcher is not supported")

        elif p == 'cosine':
            if isinstance(k, int):
                self.matcher = KNNCosineMatcher(k)
            elif k == "linear_sum":
                self.matcher = NNLinearSumCosineMatcher()
            else:
                raise ValueError("This type matcher is not supported")
        else:
            raise ValueError("This distance is not supported")

        self.model = None
        self.weighter = None

    @lal_logger.log_error
    @assertor.assert_arguments
    def fit(self, data, labels):
        """
        We use the weighter object to generate our weights, all according to which type of machine-learning task
        we are using.

        :param data: Our training data
        :type data: numpy.array
        :param labels: Our 1D training labels
        :type labels: numpy.array
        :return:
        """

        weighter = self.weighter

        if weighter is None:
            raise NotImplementedError("You need to assign a weighter object to this class to use it.")

        self.lal_logger.info("Deriving feature importance weights")
        weighter.get_feature_importances(data, labels)

        self.lal_logger.info("Constructing scaling pipeline")
        pipeline = Pipeline([("standard_scalar", StandardScaler()),
                             ("feature_scaler", _Scaler(weighter.feature_importances))])

        pipeline.fit(data, labels)

        self.model = pipeline

        return self

    def _get_matches(self, train_data, test_data):
        """
        This get the matches between the training dataset and the testing dataset using our matcher, chosen as
        initialization.

        :param train_data:
        :param test_data:
        :return:
        """

        model = self.model

        transformed_train = model.transform(train_data)
        transformed_test = model.transform(test_data)

        matcher = self.matcher

        matches = matcher.match(transformed_test, transformed_train)

        return matches

    @abstractmethod
    def predict(self, train_data, train_labels, test_data):
        pass


class LALGBClassifier(LALGBBaseModel):
    """
    This is when our training labels are categorical.
    """

    def __init__(self, k, p):
        super().__init__(k, p)

        self.weighter = LGBMClassifierWeight()

    @LALGBBaseModel.lal_logger.log_error
    @LALGBBaseModel.assertor.assert_arguments
    def predict_proba(self, train_data, train_labels, test_data):
        """
        This predicts the probability of our test data having any of the available labels in the training dataset

        :param train_data: The training dataset features
        :type train_data: numpy.array
        :param train_labels: The training dataset labels
        :type train_labels: numpy.array
        :param test_data: The testing dataset features
        :type test_data: numpy.array
        :return:
        """

        self.lal_logger.info("Scaling data and creating matches")
        matches = self._get_matches(train_data, test_data).astype(np.int64)

        # all nonzero labels
        unique_labels = np.unique(train_labels)
        num_labels = unique_labels.shape[0]

        self.lal_logger.info("Performing predictions")
        if num_labels == 2:
            preds = np.zeros((test_data.shape[0], 1))

            for i in range(test_data.shape[0]):
                preds[i, 0] = train_labels[matches[i, :]].mean()
        else:

            preds = np.zeros((test_data.shape[0], num_labels))

            multi_labels = np.zeros((train_labels.shape[0], num_labels))

            for i in range(num_labels):
                multi_labels[:, i] = (train_labels == unique_labels[i]).astype(int)

            for i in range(test_data.shape[0]):
                preds[i, :] = multi_labels[matches[i, :], :].mean(axis=0)
                preds[i, :] /= preds[i, :].sum()

        return preds

    @LALGBBaseModel.lal_logger.log_error
    @LALGBBaseModel.assertor.assert_arguments
    def predict(self, train_data, train_labels, test_data):
        """
        We choose most probable label our samples in the testing dataset has.

        :param train_data: The training dataset features
        :type train_data: numpy.array
        :param train_labels: The training dataset labels
        :type train_labels: numpy.array
        :param test_data: The testing dataset features
        :type test_data: numpy.array
        :return:
        """

        self.lal_logger.info("Generating probabilities")
        probs = self.predict_proba(train_data, train_labels, test_data)

        self.lal_logger.info("Hard-Prediction on probabilities")
        return np.argmax(probs, axis=1) + 1


class LALGBRegressor(LALGBBaseModel):
    """
    This is when our training labels are continuous.
    """

    def __init__(self, k, p):
        super().__init__(k, p)

        self.weighter = LGBMRegressorWeight()

    @LALGBBaseModel.lal_logger.log_error
    @LALGBBaseModel.assertor.assert_arguments
    def predict(self, train_data, train_labels, test_data):
        """
        We predict the possible value our testing dataset will have, based on the continuous variables.

        :param train_data: The training dataset features
        :type train_data: numpy.array
        :param train_labels: The training dataset labels
        :type train_labels: numpy.array
        :param test_data: The testing dataset features
        :type test_data: numpy.array
        :return:
        """

        self.lal_logger.info("Scaling data and creating matches")
        matches = self._get_matches(train_data, test_data).astype(np.int64)

        preds = np.zeros((test_data.shape[0], 1))

        for i in range(test_data.shape[0]):
            preds[i, 0] = train_labels[matches[i, :]].mean()

        return preds
