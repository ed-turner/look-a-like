import unittest

import logging

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

        model = GBMWeightRegressor(feature_col='v1', label_col='pred', num_folds=2, parallelism=1)

        try:
            feature_importance = model.get_feature_importances(sdf1)
        except Exception as e:
            spark.stop()

            raise e

        assert feature_importance.sum() == 1.

        assert feature_importance.min() > 0.2

        spark.stop()



if __name__ == "__main__":

    unittest.main()
