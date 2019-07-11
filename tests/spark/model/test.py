import unittest

from tests.utils.testdata.spark import PySparkTestData
from tests.utils.testframework.spark import PySparkTestFramework

from lal.spark.model import LALGBSparkRegressor, LALGBSparkMultiClassifier, LALGBSparkBinaryClassifier

from lal.utils.logger import LALLogger, CustomLogHandler


class TestLALGBSparkRegressor(PySparkTestFramework):

    logger = LALLogger("test_lal_regressor")

    @logger.log_error
    def test_predictions_1nn(self):
        self.logger.info("We are testing the weights returned sum to one and are evenly distributed")

        spark = self.spark

        if spark is None:
            raise NotImplementedError("The spark session is none!")

        generator = PySparkTestData(spark=spark)

        sdf1, sdf2 = generator.get_model_data()

        model = LALGBSparkRegressor(k=1, p=1.0, input_cols=["col_{}".format(i) for i in range(1, 5)],
                                    label_cols='pred', optimize=False)

        self.logger.info("Trying to see if fitting one model works...")
        try:
            model.fit(sdf1)
        except Exception as e:
            raise e

        pred_sdf = model.predict(sdf1, sdf2)

        self.assert_all_close_spark_dataframe(sdf1.select(["id", "pred"]).orderBy("id"),
                                              pred_sdf.select(["id", "pred"]).orderBy("id"),
                                              ["pred"], ["pred"])

        self.logger.info("SUCCESS")


class TestLALGBSparkBinaryClassifier(PySparkTestFramework):

    logger = LALLogger("test_lal_classifier")

    @logger.log_error
    def test_predictions_1nn(self):
        self.logger.info("We are testing the weights returned sum to one and are evenly distributed")

        spark = self.spark

        if spark is None:
            raise NotImplementedError("The spark session is none!")

        generator = PySparkTestData(spark=spark)

        sdf1, sdf2 = generator.get_model_data()

        model = LALGBSparkBinaryClassifier(k=1, p=1.0, input_cols=["col_{}".format(i) for i in range(1, 5)],
                                           label_cols='pred_clf', optimize=False)

        self.logger.info("Trying to see if fitting one model works...")
        try:
            model.fit(sdf1)
        except Exception as e:
            raise e

        pred_sdf = model.predict(sdf1, sdf2)

        self.assert_all_close_spark_dataframe(sdf1.select(["id", "pred_clf"]).orderBy("id"),
                                              pred_sdf.select(["id", "pred_clf"]).orderBy("id"),
                                              ["pred_clf"], ["pred_clf"])

        self.logger.info("SUCCESS")


if __name__ == "__main__":

    unittest.main()
