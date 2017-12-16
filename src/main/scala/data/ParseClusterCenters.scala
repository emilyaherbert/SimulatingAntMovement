package data

import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter

import scala.collection.mutable.Buffer

import org.apache.spark.sql.SparkSession

object ParseClusterCenters extends App {
  val spark = SparkSession.builder().master("local[*]" /*"spark://pandora00:7077"*/ ).getOrCreate()
  spark.sparkContext.setLogLevel("WARN")

  val inputFile = args(0)
  val xoutputFile = args(1)
  val youtputFile = args(2)

  val clusterCentersFile = scala.io.Source.fromFile(inputFile)
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

  val movements = Buffer[((Double, Double), (Double, Double), (Double, Double))]()
  //val deltas = Buffer[((Double, Double), (Double, Double))]()
  val xdeltas = Buffer[Array[Double]]()
  val ydeltas = Buffer[Array[Double]]()

  for (i <- 0 until clusterCenters.length - 5) {
    for (j <- 0 until clusterCenters(i).length) {
      val lookahead = Array(i, i + 1)//, i + 2, i + 3, i + 4)
        .flatMap { e =>
          val (x, y) = clusterCenters(e)(j)
          val (nx, ny) = clusterCenters(e + 1).foldLeft((100.0, (0.0, 0.0))) { (r, e) =>
            val dist = euclidianDistance(x, e._1, y, e._2)
            //if (dist < r._1) (dist -> ((e._1 - x), (e._2 - y)))
            if (dist < r._1) (dist -> e)
            else r
          }._2
          Array(nx, ny)
        }

      var kill = false
      for (i <- 0 until lookahead.length - 2) {
        if (Math.abs(lookahead(i + 2) - lookahead(i)) > 0.5) {
          kill = true
        }
      }

      if (!kill) {
        //val delta = ((nx - x) -> (ny - y), (nnx - nx) -> (nny - ny))
        //val xdelta = Array((nx - x), (ny - y), (nnx - nx))
        //val xdelta = lookahead.dropRight(1)
        val xdelta = lookahead.dropRight(1)
        xdeltas += xdelta

        lookahead(lookahead.length - 2) = lookahead(lookahead.length - 1)
        val ydelta = lookahead.dropRight(1)
        ydeltas += ydelta
        //}
      }
    }
  }

  //movements.foreach(println)
  //print(movements.length)

  //deltas.foreach(println)
  //print(deltas.length)

  val xbw = new BufferedWriter(new FileWriter(new File(xoutputFile)))
  val xsb = new StringBuilder()

  xdeltas.foreach { d =>
    xsb ++= d.mkString(",")
    xsb ++= "\n"
  }

  xbw.write(xsb.toString())
  xsb.clear()

  xbw.close()

  val ybw = new BufferedWriter(new FileWriter(new File(youtputFile)))
  val ysb = new StringBuilder()

  ydeltas.foreach { d =>
    ysb ++= d.mkString(",")
    ysb ++= "\n"
  }

  ybw.write(ysb.toString())
  ysb.clear()

  ybw.close()

  spark.stop()

  def euclidianDistance(x1: Double, x2: Double, y1: Double, y2: Double): Double = {
    Math.sqrt(Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2))
  }
}