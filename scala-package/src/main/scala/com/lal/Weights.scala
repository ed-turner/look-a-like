package com.lal

// these are some objects we expect to use as inputs and outputs of key functions
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.DenseVector

// this is for the validation split
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

// this is for the chosen model structure
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.classification.GradientBoostingClassifier
import org.apache.spark.ml.regression.GradientBoostingRegressor

// this is a private abstract class not for use on the user end side
private trait Weights{

  val featureCol: String
  val predictionCol: String

  // this paramgrid is define upon inheritance
  val paramGrid: Map[Param[_], Iterable[_]]

  // this is the base model we want to use, which is defined upon inheritance
  val model: Estimator

  def optModel(df: DataFrame): Estimator = {
    """
      | This function will optimize the Estimator chosen for this class
    """.stripMargin
    // this is pseudo code to describe what I would like to do

    // this should split the dataframe into two
    // train_df, val_df = TrainValidationSplit().split(df)

    // this should use the parameters, build
    // CrossValidator(ParamGridBuilder.build(paramGrid)).fit(train_df, val_df)

    // the optimized parameters are then parsed and set in the base model.

    // model.setParams(optParams)

  }

  // this is dependent on the base model.. But this should return a DenseVector
  def getFeatureWeights: DenseVector

}

// this class will use the GradientBoostingRegressor
case class GBRegressorWeights(featureCol: String, predictionCol: String) extends Weights {

  val paramGrid = Map(
    "learning_rate" -> Array((-15 to 15).map(x => Math.pow(10.0, x.toDouble))),
    "n_estimators" -> Array((0 until 15).map(x => 100*(x + 1)))
  )

  val model: GradientBoostingRegressor = GradientBoostingRegressor(featureCol=featureCol,
    predictionCol=predictionCol)

  def getFeatureWeights(df: DataFrame): DenseVector = {
    val opt_model: GradientBoostingRegressor = this.optModel(df)

    val fitted_model = opt_model.fit(df)

    return fitted_model.feature_importances
  }
}

// this class will use the GradientBoostingRegressor
case class GBClassifierWeights(featureCol: String, predictionCol: String) extends Weights {

  val paramGrid: Map[Param[_], Iterable[_]] = Map(
    "learning_rate" -> Array((-15 to 15).map(x => Math.pow(10.0, x.toDouble))),
    "n_estimators" -> Array((0 until 15).map(x => 100*(x + 1)))
  )

  val model: GradientBoostingClassifier = GradientBoostingClassifier(featureCol=featureCol,
    predictionCol=predictionCol)

  def getFeatureWeights(df: DataFrame): DenseVector = {
    val opt_model = this.optModel(df)

    val fitted_model = opt_model.fit(df)

    return fitted_model.feature_importances
  }
}