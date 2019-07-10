import unittest

from tests.utils.testdata.spark import PySparkTestData
from tests.utils.testframework.spark import PySparkTestFramework

from lal.spark.weights import GBMWeightBinaryClassifier, GBMWeightRegressor

from lal.utils.logger import LALLogger, CustomLogHandler


class TestGBMWeightRegressor(PySparkTestFramework):

    logger = LALLogger("test_weight_regressor", CustomLogHandler("weights.log"))

    @logger.log_error
    def test_feature_importance(self):
        self.logger.info("We are testing the weights returned sum to one and are evenly distributed")

        spark = self.spark

        if spark is None:
            raise NotImplementedError("The spark session is none!")

        generator = PySparkTestData(spark=spark)

        sdf1, sdf2 = generator.get_weight_data()

        model = GBMWeightRegressor(feature_col='v1', label_col='pred')

        self.logger.info("Trying to see if fitting one model works...")
        try:
            one_model = model.model.fit(sdf1)
        except Exception as e:
            raise e

        feat_imports_sum = one_model.featureImportances.toArray().sum()

        self.logger.info("Feature Importance Sum: {}".format(feat_imports_sum))

        assert abs(feat_imports_sum - 1.) < 1e-5

        self.logger.info("Now trying to see if we can optimize the model altogether.")
        try:
            feature_importance = model.get_feature_importances(sdf1)
        except Exception as e:
            spark.stop()

            raise e

        feat_imports_sum = feature_importance.sum()

        assert abs(feat_imports_sum - 1.) < 1e-5

        spark.stop()

        self.logger.info("SUCCESS")


class TestGBMWeightClassifier(PySparkTestFramework):

    logger = LALLogger("test_weight_classifier", CustomLogHandler("weights.log"))

    @logger.log_error
    def test_feature_importance(self):
        self.logger.info("We are testing the weights returned sum to one and are evenly distributed")

        spark = self.spark

        if spark is None:
            raise NotImplementedError("The spark session is none!")

        generator = PySparkTestData(spark=spark)

        sdf1, sdf2 = generator.get_weight_data()

        model = GBMWeightBinaryClassifier(feature_col='v1', label_col='pred_clf')

        self.logger.info("Trying to see if fitting one model works...")
        try:
            one_model = model.model.fit(sdf1)
        except Exception as e:
            raise e

        feat_imports_sum = one_model.featureImportances.toArray().sum()

        self.logger.info("Feature Importance Sum: {}".format(feat_imports_sum))

        assert abs(feat_imports_sum - 1.) < 1e-5

        self.logger.info("Now trying to see if we can optimize the model altogether.")
        try:
            feature_importance = model.get_feature_importances(sdf1)
        except Exception as e:
            spark.stop()

            raise e

        feat_imports_sum = feature_importance.sum()

        assert abs(feat_imports_sum - 1.) < 1e-5

        spark.stop()

        self.logger.info("SUCCESS")


if __name__ == "__main__":

    unittest.main()
