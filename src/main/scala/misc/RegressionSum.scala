package misc

import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.util.Random
import org.nd4j.linalg.factory.Nd4j
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.nd4j.linalg.activations.Activation
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import java.util.Collections
import org.nd4j.linalg.dataset.api.DataSet
import org.apache.spark.sql.SparkSession

/**
 * Created by Anwar on 3/15/2016.
 * An example of regression neural network for performing addition
 */
object RegressionSum extends App {
  val spark = SparkSession.builder().master("local[*]" /*"spark://pandora00:7077"*/ ).getOrCreate()
  spark.sparkContext.setLogLevel("WARN")
  import spark.implicits._

  //Random number generator seed, for reproducability
  val seed = 12345;
  //Number of iterations per minibatch
  val iterations = 1;
  //Number of epochs (full passes of the data)
  val nEpochs = 200;
  //Number of data points
  val nSamples = 1000;
  //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
  val batchSize = 100;
  //Network learning rate
  val learningRate = 0.01;
  // The range of the sample data, data in range (0-1 is sensitive for NN, you can try other ranges and see how it effects the results
  // also try changing the range along with changing the activation function
  val MIN_RANGE = 0;
  val MAX_RANGE = 3;

  val rng = new Random(seed);

  //Generate the training data
  val iterator = getTrainingData(batchSize, rng);

  //Create the network
  val numInput = 2;
  val numOutputs = 1;
  val nHidden = 10;
  val net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
    .seed(seed)
    .iterations(iterations)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .learningRate(learningRate)
    .weightInit(WeightInit.XAVIER)
    .updater(Updater.NESTEROVS) //To configure: .updater(new Nesterovs(0.9))
    .list()
    .layer(0, new DenseLayer.Builder().nIn(numInput).nOut(nHidden)
      .activation(Activation.TANH)
      .build())
    .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
      .activation(Activation.IDENTITY)
      .nIn(nHidden).nOut(numOutputs).build())
    .pretrain(false).backprop(true).build());
  net.init();
  net.setListeners(new ScoreIterationListener(1));

  //Train the network on the full data set, and evaluate in periodically
  for (i <- 0 until nEpochs) {
    iterator.reset();
    net.fit(iterator);
  }
  // Test the addition of 2 numbers (Try different numbers here)
  val input = Nd4j.create(Array(0.7, 0.2), Array(1, 2));
  val out = net.output(input, false);
  System.out.println(out);

  def getTrainingData(batchSize: Int, rand: Random): DataSetIterator = {
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
}