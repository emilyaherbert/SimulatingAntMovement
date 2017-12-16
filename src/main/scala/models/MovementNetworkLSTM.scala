package models

import scalafx.application.JFXApp
import org.apache.spark.sql.SparkSession
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import java.io.File
import org.datavec.api.split.FileSplit
import org.nd4j.linalg.activations.Activation
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.nn.conf.BackpropType
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.deeplearning4j.nn.conf.layers.GravesLSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.conf.Updater
import org.nd4j.linalg.factory.Nd4j
import scalafx.scene.canvas.Canvas
import scalafx.scene.Scene
import scalafx.animation.AnimationTimer
import scalafx.scene.paint.Color
import scalafx.scene.canvas.GraphicsContext

object MovementNetworkLSTM extends JFXApp {
  val spark = SparkSession.builder().master("local[*]" /*"spark://pandora00:7077"*/ ).getOrCreate()
  spark.sparkContext.setLogLevel("WARN")
  import spark.implicits._

  val xinputFile = "/data/BigData/students/eherbert/deltas_x_lstm.txt"
  val yinputFile = "/data/BigData/students/eherbert/deltas_y_lstm.txt"

  val numLinesToSkip = 0
  val delimiter = ','
  val batchSize = 1000
  val labelIndexFrom = 8
  val labelIndexTo = 8

  val numInputs = 8
  val lstmLayerSize = 20
  val tbpttLength = 8

  val recordReader = new CSVRecordReader(numLinesToSkip, delimiter)
  val models = Array(xinputFile, yinputFile)
    .map { f =>

      recordReader.initialize(new FileSplit(new File(f)))

      val iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndexFrom, labelIndexTo, true)
      val allData = iterator.next()
      allData.shuffle()

      val testAndTrain = allData.splitTestAndTrain(0.99)
      val trainData = testAndTrain.getTrain()
      val testData = testAndTrain.getTest()

      val neuralConf = new NeuralNetConfiguration.Builder()
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
        .learningRate(0.1)
        .seed(12345)
        .regularization(true)
        .l2(0.001)
        .weightInit(WeightInit.XAVIER)
        .updater(Updater.RMSPROP)
        .list()
        .layer(0, new GravesLSTM.Builder().nIn(numInputs).nOut(lstmLayerSize)
          .activation(Activation.TANH).build())
        .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
          .activation(Activation.TANH).build())
        .layer(2, new RnnOutputLayer.Builder(LossFunction.MSE).activation(Activation.IDENTITY)
          .nIn(lstmLayerSize).nOut(1).build())
        //.backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
        .pretrain(false).backprop(true)
        .build();

      val model = new MultiLayerNetwork(neuralConf);
      model.init();
      model.setListeners(new ScoreIterationListener(1));

      model.fit(trainData)

      //model.SevaluateRegression(testData.iterateWithMiniBatches())

      model
    }

  stage = new JFXApp.PrimaryStage {
    title = "Ants"
    scene = new Scene(600, 600) {
      fill = Color.BLACK
      val canvas = new Canvas(600, 600)
      val gc = canvas.graphicsContext2D

      content = canvas

      val ants = Array.fill(20)(new Ant(util.Random.nextInt(200 + 200), util.Random.nextInt(200 + 200), Math.random(), Math.random(), Array.fill(8)(Math.random()), 500.0))

      var lastTime = 0L
      val timer = AnimationTimer { time =>
        if (lastTime > 0) {
          val delta = (time - lastTime) / 1e9
          gc.clearRect(0, 0, 600, 600)

          ants.foreach(_.move(models(0), models(1)))
          ants.foreach(_.display(gc))
        }
        lastTime = time
      }

      timer.start()
    }
  }
}

case class Ant(var cx: Double, var cy: Double, var vx: Double, var vy: Double, var values: Array[Double], speed: Double) {

  def move(xModel: MultiLayerNetwork, yModel: MultiLayerNetwork) {
    val (nvx, nvy) = makePrediction(values, xModel, yModel)
    println(nvx, nvy)
    vx = nvx
    vy = nvy
    cx += vx * speed
    cy += vy * speed
    values = updateArray(values, vx, vy)
  }

  def display(gc: GraphicsContext) {
    gc.fill = Color.White
    gc.fillOval(cx - 5, cy - 5, 10, 10)
  }

  private def updateArray(arr: Array[Double], x: Double, y: Double): Array[Double] = {
    for (i <- 2 until arr.length) {
      arr(i - 2) = arr(i)
    }
    arr(arr.length - 2) = x
    arr(arr.length - 1) = y
    arr
  }

  private def makePrediction(values: Array[Double], xModel: MultiLayerNetwork, yModel: MultiLayerNetwork): (Double, Double) = {
    val loc = Nd4j.create(values)
    (xModel.output(loc, false).getDouble(0), yModel.output(loc, false).getDouble(0))
  }
}