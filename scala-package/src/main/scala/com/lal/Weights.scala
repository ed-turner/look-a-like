package com.lal

// these are some objects we expect to use as inputs and outputs of key functions
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.DenseVector

// this is for the validation split
import org.apache.spark.ml.PredictionModel
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
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
sealed trait Weights{

  val featureCol: String
  val predictionCol: String

  // this paramgrid is define upon inheritance
  val paramGrid: Array[ParamMap]

  // the evaluator to use
  val evaluator: Any

  // this is the base model we want to use, which is defined upon inheritance
  val model: Any

  /**
    * @return Returns the optimized `model` of this weighting process.
    */
  def optModel(df: DataFrame): TrainValidationSplitModel = {
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
  def getFeatureWeights(df: DataFrame): DenseVector = {
    val opt_model = this.optModel(df)

    val fitted_model = opt_model.transform(df)

    return fitted_model.featureImportances
  }

}

/**
  * A trait to represent a ''feature weighting process'' using gradient boosting process.
  *
  *
  * @author Edward Turner
  * @todo Add additional documentation about this trait
  * @version 1.0
  */
sealed trait GBRegWeights {

  val model = new GBTRegressor()

  val evaluator = new RegressionEvaluator()

  val paramGrid = new ParamGridBuilder()
    .addGrid(model.maxIter, Array((0 until 15).map(x => 100*(x + 1))))
    .addGrid(model.maxDepth, Array((0 until 15)))
    .addGrid(model.stepSize, Array((-15 to 0).map(x => Math.pow(10.0, x.toDouble))))
    .addGrid(model.subsamplingRate, Array((-15 to 0).map(x => Math.pow(10.0, x.toDouble))))
    .build()


}

sealed trait GBClasWeights {

  val evaulator = new BinaryClassificationEvaluator()

  val model = new GBTClassifier()

}

/**
  * A class to represent a ''feature weighting process'' using the Gradient-Boosting Tree regression task.
  *
  * {{{
  * val weighter = GBRegressorWeights("prediction", "label")
  * weighter.featureCol
  * weighter.predictionCol
  * }}}
  * @author Edward Turner
  * @todo Add additional documentation about this class
  * @version 1.0
  */
final case class GBRegressorWeights() extends Weights with GBRegWeights {

}


/**
  * A class to represent a ''feature weighting process'' using the Gradient-Boosting Tree classification task.
  *
  * {{{
  * val weighter = GBClassifierWeights("prediction", "label")
  * weighter.featureCol
  * weighter.predictionCol
  * }}}
  * @author Edward Turner
  * @todo Add additional documentation about this class
  * @version 1.0
  */
final case class GBClassifierWeights() extends Weights with GBClasWeights {


}