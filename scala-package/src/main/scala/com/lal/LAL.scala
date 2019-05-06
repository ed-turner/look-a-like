package com.lal

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineModel}

import org.apache.spark.ml.feature.{ElementwiseProduct, StandardScaler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.{Param, ParamMap}


/**
  * An abstract class to represent a ''Look-A-Like'' model.
  *
  * Specify the `predictionCol`, `labelCol` then access the fields like this:
  *
  *
  * @constructor Create a new Look-A-Like with a `predictionCol`, `labelCol`.
  * @param labelCol The person's name.
  * @param predictionCol The person's age.
  * @author Edward Turner
  * @version 1.0
  * @todo Add more functionality.
  */

abstract class LAL(predictionCol: String, labelCol: String){

  val weighter: Weights

  def fit(train_data: DataFrame): LALModel
}

/**
  * An abstract class to represent a fitted ''Look-A-Like'' model.
  *
  * Specify the `predictionCol`, `labelCol` then access the fields like this:
  *
  *
  * @constructor Create a new Look-A-Like with a `matcher`, `featureScaler`.
  * @param labelCol The person's name.
  * @param predictionCol The person's age.
  * @author Edward Turner
  * @version 1.0
  * @todo Add more functionality.
  */

abstract class LALModel(matcher: KNN, featureScaler: PipelineModel){

  def getMatches(train_data: DataFrame, test_data: DataFrame): DataFrame = matcher.knn_match(train_data, test_data)

  def transform(train_data: DataFrame, test_data: DataFrame): DataFrame = {

    this.getMatches(train_data, test_data)

  }
}

final case class LALPowerMeasureRegressor(k: BigInt, p: Double, featureCol: String, predictionCol: String) extends LAL {

  val weighter: Weights = GBRegressorWeights(featureCol, predictionCol)

  def fit(train_data: DataFrame): LALPowerMeasureRegressorModel = {

    val matcher: KNN = KNNPowerMatcher(k, featureCol, p)

    val featureWeights = weighter.getFeatureWeights(train_data)

    val model = new Pipeline().setStages(
      Array(ElementwiseProduct(scalingVec = featureWeights).setInputCol("features").setOutputCol("scaled_features"),
        StandardScaler(inputCol = "scaled_features", outputCol = "standardized_features", withStd = True,
          withMean = True)))

    LALPowerMeasureRegressorModel(matcher, model.transform(train_data), predictionCol)

  }
}

final case class LALPowerMeasureRegressorModel(
                                          matcher: KNNPowerMatcher,
                                          featureScaler: PipelineModel,
                                          predictionCol: String) extends
  LALModel(matcher, featureScaler, predictionCol)


final case class LALCosineMeasureRegressor(k: BigInt, featureCol: String, predictionCol: String) extends LAL {

  val weighter: Weights = GBRegressorWeights(featureCol, predictionCol)

  def fit(train_data: DataFrame): LALPowerMeasureRegressorModel = {

    val matcher: KNN = KNNCosine

    val featureWeights = weighter.getFeatureWeights(train_data)

    val model = new Pipeline().setStages(
      Array(ElementwiseProduct(scalingVec = featureWeights).setInputCol("features").setOutputCol("scaled_features"),
        StandardScaler(inputCol = "scaled_features", outputCol = "standardized_features", withStd = True,
          withMean = True)))

    LALPowerMeasureRegressorModel(matcher, model.transform(train_data), predictionCol)

  }
}

final case class LALCosineMeasureRegressorModel(
                                                matcher: KNNPowerMatcher,
                                                featureScaler: PipelineModel,
                                                predictionCol: String) extends
  LALModel(matcher, featureScaler, predictionCol)


