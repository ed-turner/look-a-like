package com.lal

import org.apache.spark.ml.param.Param
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

abstract private class Weights(featureCol: String, predictionCol: String, metricName: String) {

  type Space = Map[Param[_], Iterable[_]]

  val paramgrid: Space

  def optModel: Any

  def getFeatureWeights: Any

}

