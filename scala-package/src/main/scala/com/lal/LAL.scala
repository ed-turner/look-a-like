package com.lal

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.PipelineModel

abstract class LAL(k: BigInt, featureCol: String, predictionCol: String){
  abstract val weighter: Weights

  def fit(train_data: DataFrame, test_data: DataFrame): LALModel
}

abstract class LALModel(matcher: KNN, featureScaler: PipelineModel, predictionCol: String){

  def getMatches(train_data: DataFrame, test_data: DataFrame): DataFrame = matcher.knn_match(train_data, test_data)

  def transform(train_data: DataFrame, test_data: DataFrame): DataFrame
}

//
//class LALRegressor extends LAL {
//
//}
//
//class LALClassifier extends LAL {
//
//}