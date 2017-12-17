package demos

import java.io.File

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.GeneralizedLinearRegression
import org.apache.spark.ml.regression.GeneralizedLinearRegressionModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StructType
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

import scalafx.animation.AnimationTimer
import scalafx.application.JFXApp
import scalafx.scene.Scene
import scalafx.scene.canvas.Canvas
import scalafx.scene.canvas.GraphicsContext
import scalafx.scene.paint.Color
import utility.Model
import utility.NeuralNetworkModel
import utility.NormalModelInfo

object ColonySimulation extends JFXApp {
  val spark = SparkSession.builder().master("local[*]" /*"spark://pandora00:7077"*/ ).getOrCreate()
  spark.sparkContext.setLogLevel("WARN")

  val x_file = "data/deltas_x.txt"
  val y_file = "data/deltas_y.txt"
  val inputFiles = Array(x_file, y_file)

  val schema = StructType(Array(
    StructField("x", DoubleType),
    StructField("y", DoubleType),
    StructField("n", DoubleType)))

  val memorySchema = StructType(Array(
    StructField("x1", DoubleType),
    StructField("y1", DoubleType),
    StructField("x2", DoubleType),
    StructField("y2", DoubleType),
    StructField("x3", DoubleType),
    StructField("y3", DoubleType),
    StructField("x4", DoubleType),
    StructField("y4", DoubleType),
    StructField("n", DoubleType)))

  val neuralNetworkModel = NeuralNetworkModel(x_file, y_file, 0, ',', 100, 2, 2)

  val generalizedLinearRegressionModels = inputFiles.map { file =>
    val model = getGeneralizedLinearRegressionModel(file, schema)
    model
  }

  stage = new JFXApp.PrimaryStage {
    title = "Ants"
    scene = new Scene(600, 600) {
      fill = Color.BLACK
      val canvas = new Canvas(600, 600)
      val gc = canvas.graphicsContext2D

      content = canvas

      val ants = Array.fill(20)(new SimulationAnt(util.Random.nextInt(200) + 200,
        util.Random.nextInt(200) + 200,
        2000.0,
        neuralNetworkModel,
        NormalModelInfo()))

      var lastTime = 0L
      val timer = AnimationTimer { time =>
        if (lastTime > 0) {
          val delta = (time - lastTime) / 1e9
          gc.clearRect(0, 0, 600, 600)

          ants.foreach(_.move(delta))
          ants.foreach(_.display(gc))

          /*
          for (i <- 0 until 600) {
            for (j <- 0 until 600) {
              pw.setColor(i, j, Color.Black)
            }
          }
          ants.foreach(_.save(pw))
          ImageIO.write(SwingFXUtils.fromFXImage(wimg, null), "png", new java.io.File(outputFileStub + "frame_" + index + ".png"))
          index += 1
          * 
          */
        }
        lastTime = time
      }

      timer.start()
    }
  }

  spark.sparkContext.stop()

  def getGeneralizedLinearRegressionModel(file: String, schema: StructType): GeneralizedLinearRegressionModel = {
    val df = spark.read.schema(schema).option("header", false).option("delimiter", ",").csv(file)
    val va = new VectorAssembler().setInputCols(df.columns.dropRight(1)).setOutputCol("features")
    val withFeatures = va.transform(df)
    val lr = new GeneralizedLinearRegression()
      .setFamily("gaussian")
      .setLink("identity")
      .setLabelCol(df.columns.last)
    val model = lr.fit(withFeatures)
    model
  }

}