package misc

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.Level
import org.apache.log4j.Logger
import javax.imageio.ImageIO
import java.io.File

object Test extends App {
  val conf = new SparkConf().setAppName("SimulatingAntMovement-Tester").setMaster("spark://pandora00:7077")
  val sc = new SparkContext(conf)
  Logger.getLogger("org").setLevel(Level.OFF)
  sc.setLogLevel("WARN")

  val img = ImageIO.read(new File("/data/BigData/students/eherbert/frames/ants-0000001.png"))

  println("/*---------------------(づ｡◕‿‿◕｡)づ---------------------*/")
  
  println(img.toString())

  println("/*---------------------(ﾉ ｡◕‿‿◕｡)ﾉ*:･ﾟ✧ ✧ﾟ･-------------*/")

  sc.stop()
}