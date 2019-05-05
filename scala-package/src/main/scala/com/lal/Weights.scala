package com.lal

// these are some objects we expect to use as inputs and outputs of key functions
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.DenseVector

// this is for the validation split
import org.apache.spark.ml.PredictionModel
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaulation.{Evaluator, BinaryClassificationEvaluator, RegressionEvaluator}

// this is for the chosen model structure
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.regression.GBTRegressor

/**
  * A trait to represent a ''feature weighting process''.
  *
  *
  * @author Edward Turner
  * @todo Add additional documentation about this trait
  * @version 1.0
  */

// this is a private abstract class not for use on the user end side
private trait Weights{

  val featureCol: String
  val predictionCol: String

  // this paramgrid is define upon inheritance
  val paramGrid: Map[Param[_], Iterable[_]]

  // the evaluator to use
  val evaluator: Evaluator

  // this is the base model we want to use, which is defined upon inheritance
  val model: Estimator

  /**
    * @return Returns the optimized `model` of this weighting process.
    */
  def optModel(df: DataFrame): PredictionModel = {
    """
      | This function will optimize the Estimator chosen for this class
    """.stripMargin

    val cv = new TrainValidationSplit()
      .setEstimator(model)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)
      .setParallelism(2)

    cv.fit(df)

  }

  /**
    * @return Returns feature importance based on the weighting process
    */
  def getFeatureWeights: DenseVector

}

/**
  * A trait to represent a ''feature weighting process'' using gradient boosting process.
  *
  *
  * @author Edward Turner
  * @todo Add additional documentation about this trait
  * @version 1.0
  */
private trait GBWeights {

  val paramGrid = new ParamGridBuilder()
    .addGrid(model.maxIter, Array((0 until 15).map(x => 100*(x + 1))))
    .addGrid(model.maxDepth, Array((0 until 15))
      .addGrid(model.stepSize, Array((-15 to 0).map(x => Math.pow(10.0, x.toDouble))))
      .addGrid(model.subsamplingRate, Array((-15 to 0).map(x => Math.pow(10.0, x.toDouble))))
      .build()

  /**
    * @return Returns feature importance based on the weighting process normalized to sum to one.
    */
  def getFeatureWeights(df: DataFrame): DenseVector = {
    val opt_model = this.optModel(df)

    val fitted_model = opt_model.fit(df)

    return fitted_model.featureImportances
  }

}


/**
  * A class to represent a ''feature weighting process'' using the Gradient-Boosting Tree regression task.
  *
  *
  * @author Edward Turner
  * @todo Add additional documentation about this class
  * @version 1.0
  */
final case class GBRegressorWeights(featureCol: String, predictionCol: String) extends Weights with GBWeights {

  val evaluator = new RegressionEvaluator

  val model: GradientBoostingRegressor = GradientBoostingRegressor(featureCol=featureCol,
    predictionCol=predictionCol)

}


/**
  * A class to represent a ''feature weighting process'' using the Gradient-Boosting Tree classification task.
  *
  *
  * @author Edward Turner
  * @todo Add additional documentation about this class
  * @version 1.0
  */
final case class GBClassifierWeights(featureCol: String, predictionCol: String) extends Weights with GBWeights {

  val evaulator = new BinaryClassificationEvaluator

  val model: GBTClassifier = GBTClassifier(featureCol=featureCol,
    predictionCol=predictionCol)

}