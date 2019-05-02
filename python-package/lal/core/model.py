from abc import abstractmethod, ABCMeta

import numpy as np

from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .nn import KNNPowerMatcher, KNNCosineMatcher, NNLinearSumCosineMatcher, NNLinearSumPowerMatcher
from .weights import LGBMClassifierWeight, LGBMRegressorWeight


class _Scaler(TransformerMixin):
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

    def fit(self, data, labels):
        """

        :param data:
        :param labels:
        :return:
        """

        weighter = self.weighter

        if weighter is None:
            raise NotImplementedError("You need to assign a weighter object to this class to use it.")

        weighter.get_feature_importances(data, labels)

        pipeline = Pipeline([("standard_scalar", StandardScaler()),
                             ("feature_scaler", _Scaler(weighter.feature_importances))])

        pipeline.fit(data, labels)

        self.model = pipeline

        return self

    def _get_matches(self, train_data, test_data):
        """

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

    def __init__(self, k, p):
        super().__init__(k, p)

        self.weighter = LGBMClassifierWeight()

    def predict_proba(self, train_data, train_labels, test_data):
        """

        :param train_data:
        :param train_labels:
        :param test_data:
        :return:
        """

        matches = self._get_matches(train_data, test_data).astype(np.int64)

        # all nonzero labels
        unique_labels = np.unique(train_labels)
        num_labels = unique_labels.shape[0]

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

    def predict(self, train_data, train_labels, test_data):
        """

        :param train_data:
        :param train_labels:
        :param test_data:
        :return:
        """

        probs = self.predict_proba(train_data, train_labels, test_data)

        return np.argmax(probs, axis=1)


class LALGBRegressor(LALGBBaseModel):

    def __init__(self, k, p):
        super().__init__(k, p)

        self.weighter = LGBMRegressorWeight()

    def predict(self, train_data, train_labels, test_data):
        """

        :param train_data:
        :param train_labels:
        :param test_data:
        :return:
        """

        matches = self._get_matches(train_data, test_data).astype(np.int64)

        preds = np.zeros((test_data.shape[0], 1))

        for i in range(test_data.shape[0]):
            preds[i, 0] = train_labels[matches[i, :]].mean()

        return preds
