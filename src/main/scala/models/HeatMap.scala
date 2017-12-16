package models

import scalafx.application.JFXApp
import org.apache.spark.sql.SparkSession
import utility.SimilarSizeKMeans
import scalafx.scene.canvas.Canvas
import scalafx.scene.Scene
import scalafx.scene.paint.Color
import swiftvis2.plotting.ColorGradient
import swiftvis2.plotting.RedARGB
import swiftvis2.plotting.BlueARGB
import swiftvis2.plotting.GreenARGB

object HeatMap extends JFXApp {
  val spark = SparkSession.builder().master("local[*]").getOrCreate()
  spark.sparkContext.setLogLevel("WARN")

  val file = "/data/BigData/students/eherbert/clusterlocations_official.txt"
  val data = scala.io.Source.fromFile(file)
    .getLines()
    .toArray
    .map { l =>
      l.split(";").map { d =>
        val Array(x, y) = d.drop(1).dropRight(1).split(",").map { e =>
          e.toDouble.toInt
        }
        x -> y
      }
    }

  val width = 720
  val height = 480
  val map = Array.fill(width)(Array.fill(height)(0))

  for (i <- 0 until data.length) {
    for (j <- 0 until data(i).length) {
      val (x, y) = data(i)(j)
      //println(x, y)
      map(x)(y) += 1
    }
  }

  val maxElem = map.map(_.max).max
  val minElem = map.map(_.min).min
  
  println(maxElem)
  println(minElem)

  println("/*---------------------(づ｡◕‿‿◕｡)づ---------------------*/")

  stage = new JFXApp.PrimaryStage {
    title = "Heat Map"
    scene = new Scene(720, 480) {
      fill = Color.BLACK
      val canvas = new Canvas(720, 480)
      val gc = canvas.graphicsContext2D
      content = canvas

      val cg = ColorGradient((minElem, BlueARGB), ((minElem + maxElem)/4, GreenARGB), ((minElem + maxElem)/3, RedARGB))

      for (i <- 0 until map.length) {
        for (j <- 0 until map(i).length) {
          val color = cg(map(i)(j))
          gc.fill = Color.rgb((color >> 16) & 0xff, (color >> 8) & 0xff, color & 0xff)
          gc.fillRect(i, j, 1, 1)
        }
      }
    }
  }

  println("/*---------------------(ﾉ ｡◕‿‿◕｡)ﾉ*:･ﾟ✧ ✧ﾟ･-------------*/")
  spark.stop()
}