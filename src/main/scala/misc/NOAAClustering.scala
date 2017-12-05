package misc

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

import scalafx.application.JFXApp
import swiftvis2.plotting.ArrayToDoubleSeries
import swiftvis2.plotting.ArrayToIntSeries
import swiftvis2.plotting.BlueARGB
import swiftvis2.plotting.ColorGradient
import swiftvis2.plotting.DoubleToDoubleSeries
import swiftvis2.plotting.GreenARGB
import swiftvis2.plotting.Plot
import swiftvis2.plotting.RedARGB
import swiftvis2.plotting.renderer.FXRenderer

case class NOAAData(sid: String, date: java.sql.Date, measure: String, value: Double)
case class Station(sid: String, lat: Double, lon: Double, elev: Double, name: String)

object NOAAClustering extends JFXApp {
  val spark = SparkSession.builder.appName("NOAA SQL Data").master("local[*]").getOrCreate()
  import spark.implicits._

  spark.sparkContext.setLogLevel("WARN")

  //  val data2017 = spark.read.schema(Encoders.product[NOAAData].schema).option("dateFormat", "yyyyMMdd").
  //    csv("data/NOAA/2017.csv").as[NOAAData].cache()

  val stations = spark.read.textFile("../data/NOAA/ghcnd-stations.txt").map { line =>
    val id = line.substring(0, 11)
    val lat = line.substring(12, 20).trim.toDouble
    val lon = line.substring(21, 30).trim.toDouble
    val elev = line.substring(31, 37).trim.toDouble
    val name = line.substring(41, 71).trim
    Station(id, lat, lon, elev, name)
  }.cache()

  val stationVA = new VectorAssembler().setInputCols(Array("lat", "lon")).setOutputCol("location")
  val stationsWithVect = stationVA.transform(stations)

  val kMeans = new KMeans().setK(2000).setFeaturesCol("location")
  val stationClusterModel = kMeans.fit(stationsWithVect)

  val stationsWithClusters = stationClusterModel.transform(stationsWithVect)
  stationsWithClusters.show

  val x = stationsWithClusters.select('lon).as[Double].collect()
  val y = stationsWithClusters.select('lat).as[Double].collect()
  val predict = stationsWithClusters.select('prediction).as[Double].collect()
  val cg = ColorGradient(0.0 -> BlueARGB, 1000.0 -> RedARGB, 2000.0 -> GreenARGB)
  val plot = Plot.scatterPlot(x, y, "Station Clusters", "Longitude", "Latitude", 3, predict.map(cg))

  FXRenderer(plot)

  spark.stop()
}