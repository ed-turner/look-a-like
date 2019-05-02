package com.lal

abstract class Weights(featureCol: String, predictionCol: String, metricName: String) {

  val space: Any

  def optModel: Any

  def getFeatureWeights: Any

}
