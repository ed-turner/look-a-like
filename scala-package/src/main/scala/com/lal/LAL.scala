package com.lal

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.PipelineModel

abstract class LAL(predictionCol: String){

  val weighter: Weights

  def fit(train_data: DataFrame, test_data: DataFrame): LALModel
}

abstract class LALModel(matcher: KNN, featureScaler: PipelineModel){

  def getMatches(train_data: DataFrame, test_data: DataFrame): DataFrame = matcher.knn_match(train_data, test_data)

  def transform(train_data: DataFrame, test_data: DataFrame): DataFrame
}

case class LALPowerMeasureRegressor(k: BigInt, p: Double, featureCol: String, predictionCol: String) extends LAL {

  val weighter: Weights = GBRegressorWeights(featureCol, predictionCol)

  def fit(train_data, test_data): LALPowerMeasureRegressorModel = {

    val matcher: KNN = KNNPowerMatcher(k, featureCol, p)

    LALPowerMeasureRegressorModel(matcher, )

  }

case class LALPowerMeasureRegressorModel(
                                          matcher: KNNPowerMatcher,
                                          featureScaler: PipelineModel,
                                          predictionCol: String) extends
  LALModel(matcher, featureScaler, predictionCol)


//
//class LALRegressor extends LAL {
//
//}
//
//class LALClassifier extends LAL {
//
//}