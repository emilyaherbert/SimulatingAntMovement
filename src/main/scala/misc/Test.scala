package misc

import java.io.File

import scala.collection.mutable.Buffer

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StructType

import javax.imageio.ImageIO
import scalafx.Includes.tuple32JfxColor
import scalafx.application.JFXApp
import scalafx.scene.Scene
import scalafx.scene.canvas.Canvas
import scalafx.scene.paint.Color

object Test extends JFXApp {
  val spark = SparkSession.builder().master("local[*]" /*"spark://pandora00:7077"*/ ).getOrCreate()
  spark.sparkContext.setLogLevel("WARN")

  val schema = StructType(Array(
    StructField("x", IntegerType),
    StructField("y", IntegerType)))

  val numClusters = 20

  val (file, img) = imageToRead("/data/BigData/students/eherbert/frames/ants-0000101.png")
  //val (file, img) = imageToRead("/data/BigData/students/eherbert/test.png")
  val data = spark.createDataFrame(spark.sparkContext.makeRDD(file.map(Row.fromTuple(_))), schema)

  val assembler = new VectorAssembler().setInputCols(Array("x", "y")).setOutputCol("features")
  val dataWithFeatures = assembler.transform(data)

  val normalizer = new Normalizer().setInputCol("features")
  val normData = normalizer.transform(dataWithFeatures)

  val kmeans = new KMeans().setK(numClusters).setFeaturesCol("features").setPredictionCol("prediction")
  //val kmeans = new StreamingKMeans().setK(20).setFeaturesCol("features").setPredictionCol("prediction")
  val model = kmeans.fit(normData)

  val cost = model.computeCost(normData)
  val predictions = model.transform(normData)
  val centers = model.clusterCenters

  println("/*---------------------(づ｡◕‿‿◕｡)づ---------------------*/")

  data.orderBy("x").show()
  dataWithFeatures.show()
  predictions.show()
  centers.foreach(println)

  stage = new JFXApp.PrimaryStage {
    title = "Image Visualizer"
    scene = new Scene(img.length, img(0).length) {
      fill = Color.BLACK
      val canvas = new Canvas(img.length, img(0).length)
      val gc = canvas.graphicsContext2D
      content = canvas

      for (i <- 0 until img.length; j <- 0 until img(0).length) {
        val n = 255 - img(i)(j).toInt
        gc.fill = new Color(n, n, n)
        gc.fillRect(i, j, 1, 1)
      }

      //gc.fill = Color.Red
      //for (point <- centers) gc.fillOval(point(0) - 5, point(1) - 5, 10, 10)

      val colors = Array.fill(numClusters)(new Color(util.Random.nextInt(255), util.Random.nextInt(255), util.Random.nextInt(255)))
      predictions.collect().foreach { row =>
        gc.fill = colors(row.getInt(4))
        gc.fillRect(row.getInt(0), row.getInt(1), 1, 1)
      }
    }
  }

  println("/*---------------------(ﾉ ｡◕‿‿◕｡)ﾉ*:･ﾟ✧ ✧ﾟ･-------------*/")
  spark.stop()

  private def pixels2gray(red: Double, green: Double, blue: Double): Double = (red + green + blue) / 3.0

  private def imageToRead(image: String): (Array[(Int, Int)], Array[Array[Double]]) = {
    val img = ImageIO.read(new File(image))
    val arr = Array.fill(img.getWidth)(Array.fill(img.getHeight)(0.0))
    val buff = Buffer[(Int, Int)]()
    for (i <- 0 until img.getWidth; j <- 0 until img.getHeight) {
      val col = img.getRGB(i, j)
      val red = (col & 0xff0000) / 65536.0
      val green = (col & 0xff00) / 256.0
      val blue = (col & 0xff)
      val value = 255.0 - pixels2gray(red, green, blue)
      arr(i)(j) = value
      if (value > 200.0) buff += (i -> j)
    }
    (buff.toArray, arr)
  }
}