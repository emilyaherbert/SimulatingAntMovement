package data

import javax.imageio.ImageIO
import scala.collection.mutable.Buffer
import java.io.File
import java.io.FileWriter
import java.io.BufferedWriter
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DoubleType

object ReadAndSave extends App {
  val spark = SparkSession.builder().master("local[*]" /*"spark://pandora00:7077"*/ ).getOrCreate()
  spark.sparkContext.setLogLevel("WARN")

  val schema = StructType(Array(
    StructField("x", DoubleType),
    StructField("y", DoubleType)))

  val start = args(0).toInt
  val stop = args(1).toInt
  val fileSource = "/data/BigData/students/eherbert/frames/ants-"
  val outputFile = args(2)

  val numClusters = 20

  val bw = new BufferedWriter(new FileWriter(new File(outputFile),true))
  val sb = new StringBuilder()
  
  var index = 0

  for (i <- start until stop) {
    println(i)
    
    val (file, img) = imageToArray(fileSource + (i.toString.reverse + Array.fill(10)(0).mkString).take(7).reverse + ".png")
    val data = spark.createDataFrame(spark.sparkContext.makeRDD(file.map(Row.fromTuple(_))), schema)
    val assembler = new VectorAssembler().setInputCols(Array("x", "y")).setOutputCol("features")
    val dataWithFeatures = assembler.transform(data)

    val normalizer = new Normalizer().setInputCol("features")
    val normData = normalizer.transform(dataWithFeatures)

    val kmeans = new KMeans().setK(numClusters).setFeaturesCol("features").setPredictionCol("prediction")
    val model = kmeans.fit(normData)
    val centers = model.clusterCenters.map{r =>
      (r(0) -> r(1))
    }.mkString(";")
    
    sb ++= centers
    sb ++= "\n"
    
    if(index % 100 == 0) {
      bw.write(sb.toString())
      sb.clear()
    }
    
    index+=1
  }
  
  bw.write(sb.toString())
  
  bw.close()

  // file reading methods
  private def pixels2gray(red: Double, green: Double, blue: Double): Double = (red + green + blue) / 3.0

  private def imageToArray(image: String): (Array[(Double, Double)], Array[Array[Double]]) = {
    println(image)
    val img = ImageIO.read(new File(image))
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
}