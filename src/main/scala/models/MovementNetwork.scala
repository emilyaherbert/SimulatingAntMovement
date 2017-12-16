package models

import java.io.File

import org.apache.spark.sql.SparkSession
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions

import scalafx.application.JFXApp
import org.nd4j.linalg.dataset.DataSet
import scalafx.scene.canvas.Canvas
import scalafx.scene.Scene
import scalafx.animation.AnimationTimer
import scalafx.scene.paint.Color
import java.util.Collections
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import scalafx.scene.canvas.GraphicsContext
import org.deeplearning4j.nn.conf.Updater
import java.util.Random
import scalafx.scene.image.WritableImage
import scalafx.scene.image.Image
import scalafx.scene.image.PixelWriter
import javax.imageio.ImageIO
import scalafx.embed.swing.SwingFXUtils

object MovementNetwork extends JFXApp {
  val spark = SparkSession.builder().master("local[*]" /*"spark://pandora00:7077"*/ ).getOrCreate()
  spark.sparkContext.setLogLevel("WARN")
  import spark.implicits._

  val xinputFile = "/data/BigData/students/eherbert/deltas_x.txt"
  val yinputFile = "/data/BigData/students/eherbert/deltas_y.txt"

  val numLinesToSkip = 0
  val delimiter = ','
  val batchSize = 100
  val labelIndexFrom = 2
  val labelIndexTo = 2

  val seed = 12345
  val iterations = 1
  val numEpochs = 200
  val learningRate = 0.01
  val numInputs = 2
  val numOutputs = 1
  val numHiddenNodes = 10

  val recordReader = new CSVRecordReader(numLinesToSkip, delimiter)

  val models = Array((xinputFile, 2), (yinputFile, 3))
    .map { f =>

      recordReader.initialize(new FileSplit(new File(f._1)))

      val iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndexFrom, labelIndexTo, true)
      val allData = iterator.next()
      allData.shuffle()

      val testAndTrain = allData.splitTestAndTrain(0.7)
      val trainData = testAndTrain.getTrain()
      val testData = testAndTrain.getTest()

      val model = getModel()

      for (i <- 0 until numEpochs) {
        trainData.shuffle()
        model.fit(trainData)
      }

      //model.evaluateRegression(testData.i)
      println(model.score(testData))

      model
    }

  val MIN_RANGE = 0;
  val MAX_RANGE = 3;
  val rng = new Random(seed);

  val iterator = getTrainingData(batchSize, rng, 1000);
  val testData = getTrainingData(batchSize, rng, 300);
  val sumModel = getModel()

  for (i <- 0 until numEpochs) {
    iterator.reset();
    sumModel.fit(iterator);
  }

  for (i <- 0 until 10) println()

  for (i <- 0 until 10) {
    val a = Math.random().toString().take(3).toDouble
    val b = Math.random().toString().take(3).toDouble
    val input = Nd4j.create(Array(a, b), Array(1, 2));
    val out = sumModel.output(input, false);
    System.out.println(a + " + " + b + " = " + out);
  }

  println(sumModel.score(testData.next()))
  
  for (i <- 0 until 10) println()

  def getTrainingData(batchSize: Int, rand: Random, nSamples:Int): DataSetIterator = {
    val sum = Array.fill(nSamples)(0.0)
    val input1 = Array.fill(nSamples)(0.0)
    val input2 = Array.fill(nSamples)(0.0)
    for (i <- 0 until nSamples) {
      input1(i) = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * rand.nextDouble();
      input2(i) = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * rand.nextDouble();
      sum(i) = input1(i) + input2(i);
    }
    val inputNDArray1 = Nd4j.create(input1, Array(nSamples, 1));
    val inputNDArray2 = Nd4j.create(input2, Array(nSamples, 1));
    val inputNDArray = Nd4j.hstack(inputNDArray1, inputNDArray2);
    val outPut = Nd4j.create(sum, Array(nSamples, 1));
    val dataSet = new org.nd4j.linalg.dataset.DataSet(inputNDArray, outPut);
    val listDs = dataSet.asList();
    Collections.shuffle(listDs, rng);
    new ListDataSetIterator(listDs, batchSize);

  }

  val outputFileStub = "/data/BigData/students/eherbert/output/regression/"
  val wimg = new WritableImage(600, 600)
  val pw = wimg.pixelWriter
  var index = 0

  stage = new JFXApp.PrimaryStage {
    title = "Ants"
    scene = new Scene(600, 600) {
      fill = Color.BLACK
      val canvas = new Canvas(600, 600)
      val gc = canvas.graphicsContext2D

      content = canvas

      val ants = Array.fill(20)(new Ant2(util.Random.nextInt(200) + 200, util.Random.nextInt(200) + 200, Math.random()*2 - 1.0, Math.random()*2 - 1.0, 100.0))

      var lastTime = 0L
      val timer = AnimationTimer { time =>
        if (lastTime > 0) {
          val delta = (time - lastTime) / 1e9
          gc.clearRect(0, 0, 600, 600)

          ants.foreach(_.move(models(0), models(1)))
          ants.foreach(_.display(gc))

          for (i <- 0 until 600) {
            for (j <- 0 until 600) {
              pw.setColor(i, j, Color.Black)
            }
          }
          //ants.foreach(_.save(pw))
          //ImageIO.write(SwingFXUtils.fromFXImage(wimg, null), "png", new java.io.File(outputFileStub + "frame_" + index + ".png"))
          index += 1
        }
        lastTime = time
      }

      timer.start()
    }
  }

  def getModel(): MultiLayerNetwork = {
    val neuralConf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(learningRate)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.NESTEROVS)
      .list()
      .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
        .activation(Activation.TANH)
        .build())
      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .activation(Activation.IDENTITY)
        .nIn(numHiddenNodes).nOut(numOutputs).build())
      .pretrain(false).backprop(true).build();

    val model = new MultiLayerNetwork(neuralConf);
    model.init();
    model.setListeners(new ScoreIterationListener(1));

    model
  }
}

case class Ant2(var cx: Double, var cy: Double, var vx: Double, var vy: Double, speed: Double) {

  def move(xModel: MultiLayerNetwork, yModel: MultiLayerNetwork) {
    val (nvx, nvy) = makePrediction(vx, vy, xModel, yModel)
    println(nvx, nvy)
    vx = nvx
    vy = nvy
    cx += vx * speed
    cy += vy * speed
  }

  def display(gc: GraphicsContext) {
    gc.fill = Color.White
    gc.fillOval(cx - 5, cy - 5, 10, 10)
  }

  def save(pw: PixelWriter) {
    for (i <- (0 max (cx.toInt - 5)) until (600 min (cx.toInt + 5))) {
      for (j <- (0 max (cy.toInt - 5)) until (600 min (cy.toInt + 5))) {
        pw.setColor(i, j, Color.White)
      }
    }
  }

  private def updateArray(arr: Array[Double], x: Double, y: Double): Array[Double] = {
    for (i <- 2 until arr.length) {
      arr(i - 2) = arr(i)
    }
    arr(arr.length - 2) = x
    arr(arr.length - 1) = y
    arr
  }

  private def makePrediction(x: Double, y: Double, xModel: MultiLayerNetwork, yModel: MultiLayerNetwork): (Double, Double) = {
    val loc = Nd4j.create(Array(x, y), Array(1, 2))
    (xModel.output(loc, false).getDouble(0), yModel.output(loc, false).getDouble(0))
  }
}