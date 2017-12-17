package utility

import org.nd4j.linalg.dataset.DataSet
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.nd4j.linalg.activations.Activation
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.datavec.api.split.FileSplit
import java.io.File
import org.nd4j.linalg.factory.Nd4j
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.ml.regression.GeneralizedLinearRegressionModel
import org.apache.spark.ml.regression.GeneralizedLinearRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.regression.IsotonicRegression
import org.apache.spark.ml.regression.IsotonicRegressionModel

abstract class Model {
  def makePrediction(modelInfo: ModelInfo): ModelInfo
}

object SimNeuralNetworkModel {
  val seed = 12345
  val iterations = 1
  val numEpochs = 200
  val learningRate = 0.01
  val numInputs = 2
  val numOutputs = 1

  def apply(x_file: String,
    y_file: String,
    numLinesToSkip: Int,
    delimiter: Char,
    batchSize: Int,
    labelIndexFrom: Int,
    labelIndexTo: Int): SimNeuralNetworkModel = {

    val x_recordReader = new CSVRecordReader(numLinesToSkip, delimiter)
    x_recordReader.initialize(new FileSplit(new File(x_file)))

    val x_iterator = new RecordReaderDataSetIterator(x_recordReader, batchSize, labelIndexFrom, labelIndexTo, true)
    val x_allData = x_iterator.next()
    x_allData.shuffle()

    val x_testAndTrain = x_allData.splitTestAndTrain(0.7)
    val x_trainData = x_testAndTrain.getTrain()
    val x_testData = x_testAndTrain.getTest()

    val y_recordReader = new CSVRecordReader(numLinesToSkip, delimiter)
    y_recordReader.initialize(new FileSplit(new File(y_file)))

    val y_iterator = new RecordReaderDataSetIterator(y_recordReader, batchSize, labelIndexFrom, labelIndexTo, true)
    val y_allData = y_iterator.next()
    y_allData.shuffle()

    val y_testAndTrain = y_allData.splitTestAndTrain(0.7)
    val y_trainData = y_testAndTrain.getTrain()
    val y_testData = y_testAndTrain.getTest()

    new SimNeuralNetworkModel((x_trainData, x_testData),
      (y_trainData, y_testData),
      seed,
      iterations,
      learningRate,
      numInputs,
      numOutputs,
      numEpochs)
  }
}

class SimNeuralNetworkModel private (private val x_data: (DataSet, DataSet),
    private val y_data: (DataSet, DataSet),
    private val seed: Int,
    private val iterations: Int,
    private val learningRate: Double,
    private val numInputs: Int,
    private val numOutputs: Int,
    private val numEpochs: Int) extends Model {

  private val (x_testData, x_trainData) = x_data
  private val (y_testData, y_trainData) = y_data

  private val (x_model, y_model) = (buildModel(x_trainData), buildModel(y_trainData))

  private def buildModel(trainData: DataSet): MultiLayerNetwork = {
    val neuralConf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(learningRate)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.NESTEROVS)
      .list()
      .layer(0, new DenseLayer.Builder()
        .nIn(numInputs)
        .nOut(numInputs * 2)
        .activation(Activation.TANH)
        .build())
      .layer(1, new DenseLayer.Builder()
        .nIn(numInputs * 2)
        .nOut(numInputs * 4)
        .activation(Activation.TANH)
        .dropOut(0.5)
        .build())
      .layer(2, new DenseLayer.Builder()
        .nIn(numInputs * 4)
        .nOut(numInputs * 8)
        .activation(Activation.TANH)
        .dropOut(0.5)
        .build())
      .layer(3, new DenseLayer.Builder()
        .nIn(numInputs * 8)
        .nOut(numInputs * 4)
        .activation(Activation.TANH)
        .dropOut(0.5)
        .build())
      .layer(4, new DenseLayer.Builder()
        .nIn(numInputs * 4)
        .nOut(numInputs * 2)
        .activation(Activation.TANH)
        .dropOut(0.5)
        .build())
      .layer(5, new DenseLayer.Builder()
        .nIn(numInputs * 2)
        .nOut(numInputs)
        .activation(Activation.TANH)
        .dropOut(0.5)
        .build())
      .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .activation(Activation.IDENTITY)
        .nIn(numInputs)
        .nOut(numOutputs)
        .build())
      .pretrain(false).backprop(true).build();

    val model = new MultiLayerNetwork(neuralConf);
    model.init();
    model.setListeners(new ScoreIterationListener(1));

    for (i <- 0 until numEpochs) {
      trainData.shuffle()
      model.fit(x_trainData)
    }

    model
  }

  def makePrediction(modelInfo: ModelInfo): ModelInfo = {
    val loc = Nd4j.create(Array(modelInfo.vx, modelInfo.vy), Array(1, 2))
    modelInfo.update(x_model.output(loc, false).getDouble(0), y_model.output(loc, false).getDouble(0))
  }
}

object SimGeneralizedLinearRegressionModel {
  val family = "gaussian"
  val link = "identity"

  def apply(x_file: String, y_file: String, schema: StructType, spark: SparkSession): SimGeneralizedLinearRegressionModel = {

    val x_df = spark.read.schema(schema).option("header", false).option("delimiter", ",").csv(x_file).cache()
    val va = new VectorAssembler().setInputCols(x_df.columns.dropRight(1)).setOutputCol("features")
    val x_withFeatures = va.transform(x_df)

    val y_df = spark.read.schema(schema).option("header", false).option("delimiter", ",").csv(y_file).cache()
    val y_withFeatures = va.transform(y_df)

    new SimGeneralizedLinearRegressionModel(x_withFeatures, y_withFeatures, family, link, x_df.columns.last, spark)
  }
}

class SimGeneralizedLinearRegressionModel private (private val x_df: org.apache.spark.sql.Dataset[Row],
    private val y_df: org.apache.spark.sql.Dataset[Row],
    private val family: String,
    private val link: String,
    private val labelCol: String,
    private val spark: SparkSession) extends Model {
  import spark.implicits._

  private val (x_model, y_model) = (buildModel(x_df), buildModel(y_df))

  private def buildModel(df: org.apache.spark.sql.Dataset[Row]): GeneralizedLinearRegressionModel = {
    new GeneralizedLinearRegression()
      .setFamily(family)
      .setLink(link)
      .setLabelCol(labelCol)
      .fit(df)
  }

  def makePrediction(modelInfo: ModelInfo): ModelInfo = {
    val df = Seq((modelInfo.vx, modelInfo.vy)).toDF("x", "y")
    val va = new VectorAssembler().setInputCols(df.columns).setOutputCol("features")
    val withFeatures = va.transform(df)
    //withFeatures.show()
    modelInfo.update(x_model.transform(withFeatures).select("prediction").collect().head.getDouble(0), y_model.transform(withFeatures).select("prediction").collect().head.getDouble(0))
  }

  def printAccuracy() {

  }
}

object SimDecisionTreeRegressionModel {
  def apply(x_file: String, y_file: String, schema: StructType, spark: SparkSession): SimDecisionTreeRegressionModel = {
    val x_df = spark.read.schema(schema).option("header", false).option("delimiter", ",").csv(x_file).cache()
    val va = new VectorAssembler().setInputCols(x_df.columns.dropRight(1)).setOutputCol("features")
    val x_withFeatures = va.transform(x_df)
    val Array(x_trainData, x_testData) = x_withFeatures.randomSplit(Array(0.7, 0.3))

    val y_df = spark.read.schema(schema).option("header", false).option("delimiter", ",").csv(y_file).cache()
    val y_withFeatures = va.transform(y_df)
    val Array(y_trainData, y_testData) = y_withFeatures.randomSplit(Array(0.7, 0.3))

    new SimDecisionTreeRegressionModel((x_trainData, x_testData), (y_trainData, y_testData), x_df.columns.last, spark)
  }
}

class SimDecisionTreeRegressionModel private (x_data: (org.apache.spark.sql.Dataset[Row], org.apache.spark.sql.Dataset[Row]),
    y_data: (org.apache.spark.sql.Dataset[Row], org.apache.spark.sql.Dataset[Row]),
    labelCol: String,
    spark: SparkSession) extends Model {
  import spark.implicits._

  private val (x_testData, x_trainData) = x_data
  private val (y_testData, y_trainData) = y_data

  private val (x_model, y_model) = (buildModel(x_trainData), buildModel(y_trainData))

  private def buildModel(trainData: org.apache.spark.sql.Dataset[Row]): PipelineModel = {
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(trainData)
    val dtr = new RandomForestRegressor()
      .setLabelCol(labelCol)
      .setFeaturesCol("indexedFeatures")
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, dtr))
    pipeline.fit(trainData)
  }

  def makePrediction(modelInfo: ModelInfo): ModelInfo = {
    val df = Seq((modelInfo.vx, modelInfo.vy)).toDF("x", "y")
    val va = new VectorAssembler().setInputCols(df.columns).setOutputCol("features")
    val withFeatures = va.transform(df)
    //withFeatures.show()
    modelInfo.update(x_model.transform(withFeatures).select("prediction").collect().head.getDouble(0), y_model.transform(withFeatures).select("prediction").collect().head.getDouble(0))
  }

  def evaluate(): (Double, Double) = {
    val evaluator = new RegressionEvaluator()
      .setLabelCol(labelCol)
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    (evaluator.evaluate(x_model.transform(x_testData)), evaluator.evaluate(y_model.transform(y_testData)))
  }
}

object SimRandomForestRegressionModel {
  def apply(x_file: String, y_file: String, schema: StructType, spark: SparkSession): SimRandomForestRegressionModel = {
    val x_df = spark.read.schema(schema).option("header", false).option("delimiter", ",").csv(x_file).cache()
    val va = new VectorAssembler().setInputCols(x_df.columns.dropRight(1)).setOutputCol("features")
    val x_withFeatures = va.transform(x_df)
    val Array(x_trainData, x_testData) = x_withFeatures.randomSplit(Array(0.7, 0.3))

    val y_df = spark.read.schema(schema).option("header", false).option("delimiter", ",").csv(y_file).cache()
    val y_withFeatures = va.transform(y_df)
    val Array(y_trainData, y_testData) = y_withFeatures.randomSplit(Array(0.7, 0.3))

    new SimRandomForestRegressionModel((x_trainData, x_testData), (y_trainData, y_testData), x_df.columns.last, spark)
  }
}

class SimRandomForestRegressionModel private (x_data: (org.apache.spark.sql.Dataset[Row], org.apache.spark.sql.Dataset[Row]),
    y_data: (org.apache.spark.sql.Dataset[Row], org.apache.spark.sql.Dataset[Row]),
    labelCol: String,
    spark: SparkSession) extends Model {
  import spark.implicits._

  private val (x_testData, x_trainData) = x_data
  private val (y_testData, y_trainData) = y_data

  private val (x_model, y_model) = (buildModel(x_trainData), buildModel(y_trainData))

  private def buildModel(trainData: org.apache.spark.sql.Dataset[Row]): PipelineModel = {
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(trainData)
    val dtr = new DecisionTreeRegressor()
      .setLabelCol(labelCol)
      .setFeaturesCol("indexedFeatures")
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, dtr))
    pipeline.fit(trainData)
  }

  def makePrediction(modelInfo: ModelInfo): ModelInfo = {
    val df = Seq((modelInfo.vx, modelInfo.vy)).toDF("x", "y")
    val va = new VectorAssembler().setInputCols(df.columns).setOutputCol("features")
    val withFeatures = va.transform(df)
    //withFeatures.show()
    modelInfo.update(x_model.transform(withFeatures).select("prediction").collect().head.getDouble(0), y_model.transform(withFeatures).select("prediction").collect().head.getDouble(0))
  }

  def evaluate(): (Double, Double) = {
    val evaluator = new RegressionEvaluator()
      .setLabelCol(labelCol)
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    (evaluator.evaluate(x_model.transform(x_testData)), evaluator.evaluate(y_model.transform(y_testData)))
  }
}

object SimGradientBoostedTreeRegressionModel {
  def apply(x_file: String, y_file: String, schema: StructType, spark: SparkSession): SimGradientBoostedTreeRegressionModel = {
    val x_df = spark.read.schema(schema).option("header", false).option("delimiter", ",").csv(x_file).cache()
    val va = new VectorAssembler().setInputCols(x_df.columns.dropRight(1)).setOutputCol("features")
    val x_withFeatures = va.transform(x_df)
    val Array(x_trainData, x_testData) = x_withFeatures.randomSplit(Array(0.7, 0.3))

    val y_df = spark.read.schema(schema).option("header", false).option("delimiter", ",").csv(y_file).cache()
    val y_withFeatures = va.transform(y_df)
    val Array(y_trainData, y_testData) = y_withFeatures.randomSplit(Array(0.7, 0.3))

    new SimGradientBoostedTreeRegressionModel((x_trainData, x_testData), (y_trainData, y_testData), x_df.columns.last, spark)
  }
}

class SimGradientBoostedTreeRegressionModel private (x_data: (org.apache.spark.sql.Dataset[Row], org.apache.spark.sql.Dataset[Row]),
    y_data: (org.apache.spark.sql.Dataset[Row], org.apache.spark.sql.Dataset[Row]),
    labelCol: String,
    spark: SparkSession) extends Model {
  import spark.implicits._

  private val (x_testData, x_trainData) = x_data
  private val (y_testData, y_trainData) = y_data

  private val (x_model, y_model) = (buildModel(x_trainData), buildModel(y_trainData))

  private def buildModel(trainData: org.apache.spark.sql.Dataset[Row]): PipelineModel = {
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(trainData)
    val dtr = new GBTRegressor()
      .setLabelCol(labelCol)
      .setFeaturesCol("indexedFeatures")
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, dtr))
    pipeline.fit(trainData)
  }

  def makePrediction(modelInfo: ModelInfo): ModelInfo = {
    val df = Seq((modelInfo.vx, modelInfo.vy)).toDF("x", "y")
    val va = new VectorAssembler().setInputCols(df.columns).setOutputCol("features")
    val withFeatures = va.transform(df)
    //withFeatures.show()
    modelInfo.update(x_model.transform(withFeatures).select("prediction").collect().head.getDouble(0), y_model.transform(withFeatures).select("prediction").collect().head.getDouble(0))
  }

  def evaluate(): (Double, Double) = {
    val evaluator = new RegressionEvaluator()
      .setLabelCol(labelCol)
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    (evaluator.evaluate(x_model.transform(x_testData)), evaluator.evaluate(y_model.transform(y_testData)))
  }
}

object SimIsotonicRegressionModel {
  def apply(x_file: String, y_file: String, schema: StructType, spark: SparkSession): SimIsotonicRegressionModel = {
    val x_df = spark.read.schema(schema).option("header", false).option("delimiter", ",").csv(x_file).cache()
    val va = new VectorAssembler().setInputCols(x_df.columns.dropRight(1)).setOutputCol("features")
    val x_withFeatures = va.transform(x_df)
    val Array(x_trainData, x_testData) = x_withFeatures.randomSplit(Array(0.7, 0.3))

    val y_df = spark.read.schema(schema).option("header", false).option("delimiter", ",").csv(y_file).cache()
    val y_withFeatures = va.transform(y_df)
    val Array(y_trainData, y_testData) = y_withFeatures.randomSplit(Array(0.7, 0.3))

    new SimIsotonicRegressionModel((x_trainData, x_testData), (y_trainData, y_testData), x_df.columns.last, spark)
  }
}

class SimIsotonicRegressionModel private (x_data: (org.apache.spark.sql.Dataset[Row], org.apache.spark.sql.Dataset[Row]),
    y_data: (org.apache.spark.sql.Dataset[Row], org.apache.spark.sql.Dataset[Row]),
    labelCol: String,
    spark: SparkSession) extends Model {
  import spark.implicits._

  private val (x_testData, x_trainData) = x_data
  private val (y_testData, y_trainData) = y_data

  private val (x_model, y_model) = (buildModel(x_trainData), buildModel(y_trainData))

  private def buildModel(trainData: org.apache.spark.sql.Dataset[Row]): IsotonicRegressionModel = {
    new IsotonicRegression()
      .setLabelCol(labelCol)
      .setFeaturesCol("features")
      .fit(trainData)
  }

  def makePrediction(modelInfo: ModelInfo): ModelInfo = {
    val df = Seq((modelInfo.vx, modelInfo.vy)).toDF("x", "y")
    val va = new VectorAssembler().setInputCols(df.columns).setOutputCol("features")
    val withFeatures = va.transform(df)
    //withFeatures.show()
    modelInfo.update(x_model.transform(withFeatures).select("prediction").collect().head.getDouble(0), y_model.transform(withFeatures).select("prediction").collect().head.getDouble(0))
  }

  def evaluate(): (Double, Double) = {
    val evaluator = new RegressionEvaluator()
      .setLabelCol(labelCol)
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    (evaluator.evaluate(x_model.transform(x_testData)), evaluator.evaluate(y_model.transform(y_testData)))
  }
}