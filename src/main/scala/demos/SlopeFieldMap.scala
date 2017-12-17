package demos

import org.apache.spark.sql.SparkSession
import scala.collection.mutable.Buffer
import scalafx.application.JFXApp
import swiftvis2.plotting.ColorGradient
import scalafx.scene.canvas.Canvas
import scalafx.scene.Scene
import scalafx.scene.paint.Color

object SlopeFieldMap extends JFXApp {
  val spark = SparkSession.builder().master("local[*]").getOrCreate()
  spark.sparkContext.setLogLevel("WARN")

  val clusterCentersFile = scala.io.Source.fromFile("/data/BigData/students/eherbert/clusterlocations_official.txt")
  val clusterCenters = clusterCentersFile
    .getLines()
    .toArray
    .map { l =>
      l.split(";").map { d =>
        val Array(x, y) = d.drop(1).dropRight(1).split(",")
        (x.toDouble -> y.toDouble)
      }
    }
  clusterCentersFile.close()

  val slopes = Buffer[((Double, Double), (Double, Double))]()

  for (i <- 0 until clusterCenters.length - 5) {
    for (j <- 0 until clusterCenters(i).length) {
      val (x, y) = clusterCenters(i)(j)
      val (vx, vy) = clusterCenters(i + 1).foldLeft((100.0, (0.0, 0.0))) { (r, e) =>
        val dist = euclidianDistance(x, e._1, y, e._2)
        if (dist < r._1) (dist -> ((e._1 - x), (e._2 - y)))
        else r
      }._2

      if (vx < 0.3 && vy < 0.3) {
        //val vDist = Math.sqrt(Math.pow(vx, 2) + Math.pow(vy, 2))
        //val (nx,ny) = (vx/vDist,vy/vDist)
        val slope = (x, y) -> (vx, vy)
        slopes += slope
      }
    }
  }

  val width = 720
  val height = 480
  val blockSize = 15
  val numXBins = width / blockSize
  val numYBins = height / blockSize
  var map = Array.fill(numXBins)(Array.fill(numYBins)((0.0, 0.0)))

  for (i <- 0 until slopes.length) {
    val (x, y) = slopes(i)._1
    map(x.toInt / blockSize)(y.toInt / blockSize) = (map(x.toInt / blockSize)(y.toInt / blockSize)._1 + slopes(i)._2._1) -> (map(x.toInt / blockSize)(y.toInt / blockSize)._2 + slopes(i)._2._2)
  }

  map = map.map { r =>
    r.map { d =>
      val (vx, vy) = d
      val vDist = Math.sqrt(Math.pow(vx, 2) + Math.pow(vy, 2))
      (vx / vDist, vy / vDist)
    }
  }

  println("/*---------------------(づ｡◕‿‿◕｡)づ---------------------*/")

  stage = new JFXApp.PrimaryStage {
    title = "Slope Field Map"
    scene = new Scene(720, 480) {
      fill = Color.BLACK
      val canvas = new Canvas(720, 480)
      val gc = canvas.graphicsContext2D
      content = canvas

      gc.fill = Color.White
      gc.stroke = Color.White
      for (i <- 0 until map.length) {
        for (j <- 0 until map(i).length) {
          val (vx, vy) = map(i)(j)
          //println(vx, vy)
          gc.strokeLine((i * blockSize) + (blockSize / 2) + (vx / 2) * blockSize,
            (j * blockSize) + (blockSize / 2) + (vy / 2) * blockSize,
            (i * blockSize) + (blockSize / 2) - (vx / 2) * blockSize,
            (j * blockSize) + (blockSize / 2) - (vy / 2) * blockSize)
        }
      }
    }
  }

  println("/*---------------------(ﾉ ｡◕‿‿◕｡)ﾉ*:･ﾟ✧ ✧ﾟ･-------------*/")
  spark.stop()

  def euclidianDistance(x1: Double, x2: Double, y1: Double, y2: Double): Double = {
    Math.sqrt(Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2))
  }
}