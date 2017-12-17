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

abstract class Model {
  def makePrediction(modelInfo: ModelInfo): ModelInfo
}

object NeuralNetworkModel {
  val seed = 12345
  val iterations = 1
  val numEpochs = 200
  val learningRate = 0.01
  val numInputs = 2
  val numOutputs = 1

  def apply(x_file: String,
      y_file: String,
      numLinesToSkip:Int,
      delimiter:Char,
      batchSize:Int,
      labelIndexFrom:Int,
      labelIndexTo:Int): NeuralNetworkModel = {
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
    
    new NeuralNetworkModel((x_trainData, x_testData),
      (y_trainData, y_testData),
      seed,
      iterations,
      learningRate,
      numInputs,
      numOutputs,
      numEpochs)
  }
}

class NeuralNetworkModel private (private val x_data: (DataSet, DataSet),
    private val y_data: (DataSet, DataSet),
    private val seed: Int,
    private val iterations: Int,
    private val learningRate: Double,
    private val numInputs: Int,
    private val numOutputs: Int,
    private val numEpochs: Int) extends Model {

  private val (x_testData, x_trainData) = x_data
  private val (y_testData, y_trainData) = y_data

  private val (x_model, y_model) = (buildNetwork(x_trainData), buildNetwork(y_trainData))

  private def buildNetwork(trainData: DataSet): MultiLayerNetwork = {
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
        .nOut(10)
        .activation(Activation.TANH)
        .build())
      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .activation(Activation.IDENTITY)
        .nIn(10)
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
    modelInfo match {
      case n: NormalModelInfo => {
        val loc = Nd4j.create(Array(n.vx, n.vy), Array(1, 2))
        modelInfo.update(x_model.output(loc, false).getDouble(0), y_model.output(loc, false).getDouble(0))
      }
      case _ => modelInfo
    }
  }
}