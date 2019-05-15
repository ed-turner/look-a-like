from abc import abstractmethod, ABCMeta

from sklearn.model_selection import train_test_split
from sklearn.metrics import get_scorer

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from lightgbm import LGBMClassifier, LGBMRegressor


class LGBMWeightsBase(metaclass=ABCMeta):
    """
    This is our base class to help generate our importance weights for our features by using the feature_importance
    from the gradient boosting method.
    """
    def __init__(self, metric_name):
        """

        :param metric_name:
        """
        self._metric = metric_name

        self.feature_importances = None
        pass

    @staticmethod
    def _get_parameter_space():
        """
        This is our space parameter space.
        :return:
        """
        space = [Integer(1, 10, name='max_depth'),
                 Integer(100, 1000, name="n_estimators"),
                 Real(10 ** -5, 10 ** 0, "log-uniform", name='learning_rate'),
                 Real(0.1, 1.0, name="subsample"),
                 Real(0.1, 1.0, name="colsample_bytree"),
                 Real(10 ** -4.0, 10 ** 4.0, "log-uniform", name="reg_alpha"),
                 Real(10 ** -4.0, 10 ** 4.0, "log-uniform", name="reg_lambda"),
                 Integer(2, 100, name='min_child_samples')]
        return space

    @abstractmethod
    def _get_base_model(self):
        pass

    def _opt_model(self, data, labels):
        """
        This uses our chosen metric, base machine-learning-task-based model and the parameter space to optimize our
        model when fitting onto the train-split of our data and predicting on our validation-split of our data.

        :param data: Our dataset
        :param labels: Our labels
        :return:
        """

        space = self._get_parameter_space()

        metric = self._metric

        model = self._get_base_model()

        train_data, val_data, train_labels, val_labels = train_test_split(data, labels)

        funct_scorer = get_scorer(metric)

        if metric == "neg_log_loss":
            @use_named_args(space)
            def objective(**params):
                model.set_params(**params)

                model.fit(train_data, train_labels)

                val_preds = model.predict_proba(val_data)

                return funct_scorer(val_labels, val_preds)
        else:
            @use_named_args(space)
            def objective(**params):
                model.set_params(**params)

                model.fit(train_data, train_labels)

                val_preds = model.predict(val_data)

                return funct_scorer(val_labels, val_preds)

        res_gp = gp_minimize(objective, space, n_calls=50, random_state=0)

        x_sol = res_gp.get('x')

        opt_params = {space[i].name: x_sol[i] for i in range(x_sol.shape[0])}

        return opt_params

    def get_feature_importances(self, data, labels):
        """
        After optimizing the model, we fit on the whole dataset, parse the feature_importance attribute and scale it
        to sum to one.

        :param data: Our training data
        :type data: numpy.array
        :param labels: Our 1D training labels
        :type labels: numpy.array
        :return:
        """
        opt_params = self._opt_model(data, labels)

        model = self._get_base_model()

        model.set_params(**opt_params)

        model.fit(data, labels)

        feature_importances = model.feature_importances_.abs()

        feature_importances /= feature_importances.sum()

        self.feature_importances = feature_importances

        return self


class LGBMClassifierWeight(LGBMWeightsBase):
    """
    This is for our classification-task
    """
    def __init__(self):
        super().__init__("neg_log_loss")

    def _get_base_model(self):
        return LGBMClassifier()


class LGBMRegressorWeight(LGBMWeightsBase):
    """
    This is for our regression-task.
    """
    def __init__(self):
        super().__init__("neg_mean_squared_error")

    def _get_base_model(self):
        return LGBMRegressor()

