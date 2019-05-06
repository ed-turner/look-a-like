package com.lal

// this is to help parallelize the distance calculation
import scala.collection.parallel.immutable.ParVector

// this is to help define a structure for the kernel matrix
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Matrix
import org.apache.spark.sql.DataFrame

/**
  * A class to represent a ''human being''.
  *
  * Specify the `name`, `age`, and `weight` when creating a new `Person`,
  * then access the fields like this:

  * {{{
  * val p = Person("Al", 42, 200.0)
  * p.name
  * p.age
  * p.weight
  * }}}
  *
  * Did you know: The [[com.acme.foo.Employee]] extends this class.
  *
  * @constructor Create a new person with a `name`, `age`, and `weight`.
  * @param name The person's name.
  * @param age The person's age.
  * @param weight The person's weight.
  * @author Alvin Alexander
  * @version 1.0
  * @todo Add more functionality.
  * @see See [[http://alvinalexander.com alvinalexander.com]] for more "
  * information.
  */

// this is a distance trait to help create an abstract level between the KNN Method and the possible distance metrics
trait DistanceBase {

  type Sample = (BigInt, Vector)

  def calculateSampleDistance(samp1: Sample, samp2: Sample): Double

}

// this is the power distance measure
trait PowerDistance extends DistanceBase {

  val p: Double

  def calculateSampleDistance(samp1: Sample, samp2: Sample): Double = {

    val num_features: Int = samp1._2.size

    def tmp_funct(i: Int, acc: Double): Double = i match {
      case _ if i == num_features => acc
      case _ => tmp_funct(i + 1, acc + Math.pow(Math.abs(samp1._2(i) - samp2._2(i)), p))
    }

    Math.pow(tmp_funct(0, 0.0), 1.0 / p)
  }

}

// this is the knn matching algorithm, with the distance measure abstract
abstract class KNN(k: BigInt, featureCol: String) extends DistanceBase {

  def getKNeighbors(sample1: Sample, samples: Stream[Sample]): Stream[BigInt] = {
    val distances: ParVector[(BigInt, Double)] = samples.par.map(_ =>(_._1, calculateSampleDistance(sample1, _)))

    distances.toStream.sortWith(_ < _).take(k).map(_ => _._1)

  }

  def knn_match(train_data: DataFrame, test_data: DataFrame): DataFrame

}


// this is the K-Nearest Neighbors Algorithm with the Power Distance Measure
case class KNNPowerMatcher(k: BigInt, featureCol: String, p: Double) extends KNN(k, featureCol) with PowerDistance