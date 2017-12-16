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
import utility.SimilarSizeKMeans
import org.apache.spark.sql.types.DoubleType

import org.apache.spark.sql.functions._

object Test extends JFXApp {
  val spark = SparkSession.builder().master("local[*]").getOrCreate()
  spark.sparkContext.setLogLevel("WARN")

  val schema = StructType(Array(
    StructField("x", DoubleType),
    StructField("y", DoubleType)))

  val numClusters = 20

  val images = "/data/BigData/students/eherbert/frames/ants-0000001.png".split(",")
  val imageArrays = images.map(imageToArray(_))
  val data = imageArrays.map { r =>
    spark.createDataFrame(spark.sparkContext.makeRDD(r._1.map(Row.fromTuple(_))), schema)
  }
  val centers = data.map(findCentersKMeans(_))

  println("/*---------------------(づ｡◕‿‿◕｡)づ---------------------*/")

  stage = new JFXApp.PrimaryStage {
    title = "Image Visualizer"
    val sampleImg = imageArrays(0)._2
    val sampleCenter = centers(0)
    scene = new Scene(sampleImg.length, sampleImg(0).length) {
      fill = Color.BLACK
      val canvas = new Canvas(sampleImg.length, sampleImg(0).length)
      val gc = canvas.graphicsContext2D
      content = canvas

      // draw image
      for (i <- 0 until sampleImg.length; j <- 0 until sampleImg(0).length) {
        val n = 255 - sampleImg(i)(j).toInt
        gc.fill = new Color(n, n, n)
        gc.fillRect(i, j, 1, 1)
      }

      val sskmeans = new SimilarSizeKMeans(numClusters, "features", "prediction", 2.0, 1.0)
      val sampleData = data(0)

      // draw cluster centers
      gc.fill = Color.Red
      for (point <- sampleCenter) {
        gc.fillOval(point._1 - 5, point._2 - 5, 10, 10)
      }

      /*
      val colors = Array.fill(numClusters)(new Color(util.Random.nextInt(255), util.Random.nextInt(255), util.Random.nextInt(255)))
      predictions(0).collect().foreach { row =>
        gc.fill = colors(row.getInt(4))
        gc.fillRect(row.getInt(0), row.getInt(1), 1, 1)
      }
      * 
      */
    }
  }

  println("/*---------------------(ﾉ ｡◕‿‿◕｡)ﾉ*:･ﾟ✧ ✧ﾟ･-------------*/")
  spark.stop()

  // file reading methods
  private def pixels2gray(red: Double, green: Double, blue: Double): Double = (red + green + blue) / 3.0

  private def imageToArray(image: String): (Array[(Double, Double)], Array[Array[Double]]) = {
    val img = ImageIO.read(new File(image))
    println(img.getWidth)
    println(img.getHeight)
    val arr = Array.fill(img.getWidth)(Array.fill(img.getHeight)(0.0))
    val buff = Buffer[(Double, Double)]()
    for (i <- 0 until img.getWidth; j <- 0 until img.getHeight) {
      val col = img.getRGB(i, j)
      val red = (col & 0xff0000) / 65536.0
      val green = (col & 0xff00) / 256.0
      val blue = (col & 0xff)
      val value = 255.0 - pixels2gray(red, green, blue)
      arr(i)(j) = value
      if (value > 220.0) buff += (i.toDouble -> j.toDouble)
    }
    (buff.toArray, arr)
  }

  private def findCenters(data: org.apache.spark.sql.Dataset[Row]): Array[(Double, Double)] = {
    val sskmeans = new SimilarSizeKMeans(numClusters, "features", "prediction", 2.0, 2.0)
    sskmeans.findClusterCenters(data)
  }

  private def findCentersKMeans(data: org.apache.spark.sql.Dataset[Row]): Array[(Double, Double)] = {
    val assembler = new VectorAssembler().setInputCols(Array("x", "y")).setOutputCol("features")
    val dataWithFeatures = assembler.transform(data)
    val kmeans = new KMeans().setK(numClusters).setFeaturesCol("features").setPredictionCol("prediction")
    val model = kmeans.fit(dataWithFeatures)
    model.clusterCenters.map{c =>
      c(0) -> c(1)
    }
  }
}