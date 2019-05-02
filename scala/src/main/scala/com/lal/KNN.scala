package com.lal

import scala.collection.parallel.immutable.ParVector

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Matrix
import org.apache.spark.sql.DataFrame

abstract class KNN(k: Int, featureCol: String) {

  type Sample = (BigInt, Vector)

  abstract def calculateSampleDistance(samp1: Sample, samp2: Sample): Double

  def getKNeighbors(sample1: Sample, samples: Stream[Sample]): Stream[BigInt] = {
    val distances: ParVector[(BigInt, Double)] = samples.par.map(_ =>(_._1, calculateSampleDistance(sample1, _)))

    distances.toStream.sortWith(_ < _).take(k).map(_ => _._1)

  }

  def knn_match(train_data: DataFrame, test_data: DataFrame): DataFrame

}
