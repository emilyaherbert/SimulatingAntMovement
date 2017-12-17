package demos

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StructType

import javax.imageio.ImageIO
import scalafx.animation.AnimationTimer
import scalafx.application.JFXApp
import scalafx.embed.swing.SwingFXUtils
import scalafx.scene.Scene
import scalafx.scene.canvas.Canvas
import scalafx.scene.image.WritableImage
import scalafx.scene.paint.Color
import utility.NormalModelInfo
import utility.SimDecisionTreeRegressionModel
import utility.SimGeneralizedLinearRegressionModel
import utility.SimNeuralNetworkModel
import utility.SimRandomForestRegressionModel
import utility.SimGradientBoostedTreeRegressionModel
import utility.SimIsotonicRegressionModel


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

  val nn = SimNeuralNetworkModel(x_file, y_file, 0, ',', 100, 2, 2)
  val glr = SimGeneralizedLinearRegressionModel(x_file, y_file, schema, spark)
  val dtr = SimDecisionTreeRegressionModel(x_file, y_file, schema, spark)
  val rfr = SimRandomForestRegressionModel(x_file, y_file, schema, spark)
  val gbtr = SimGradientBoostedTreeRegressionModel(x_file, y_file, schema, spark)
  val ir = SimIsotonicRegressionModel(x_file, y_file, schema, spark)
  
  //glrm.printAccuracy()
  println("   Generalized Linear Regression: " + glr.evaluate)
  println("       Decision Tree  Regression: " + dtr.evaluate)
  println("       Random Forest  Regression: " + rfr.evaluate)
  println("Gradient Boosted Tree Regression: " + gbtr.evaluate)
  println("             Isotonic Regression: " + ir.evaluate)

  /*
  val outputFileStub = "/data/BigData/students/eherbert/output/isotonic/"
  val wimg = new WritableImage(600, 600)
  val pw = wimg.pixelWriter
  var index = 100
  * 
  */

  stage = new JFXApp.PrimaryStage {
    title = "Ants"
    scene = new Scene(600, 600) {
      fill = Color.BLACK
      val canvas = new Canvas(600, 600)
      val gc = canvas.graphicsContext2D

      content = canvas

      val ants = Array.fill(20)(new SimulationAnt(util.Random.nextInt(200) + 200,
        util.Random.nextInt(200) + 200,
        3500.0,
        gbtr, // change this to change current sim model
        NormalModelInfo())) // change this to change if using memory

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

  //spark.sparkContext.stop()
}