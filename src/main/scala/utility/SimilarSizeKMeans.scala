package utility

import org.apache.spark.sql.Row
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._

class SimilarSizeKMeans(private val numClusters: Int,
    private val featuresCol: String,
    private val predictionCol: String,
    private val stdDevBreakpoint: Double,
    private val stdDevTolerance: Double) extends Serializable {

  def findClusterCenters(data: Dataset[Row]): Array[(Double, Double)] = {
    var startingClusterCenters = data
      .orderBy(rand())
      .take(20)
      .map { r =>
        r.getDouble(0) -> r.getDouble(1)
      }.zipWithIndex

    // wow i just really hate everything about this
    var (clusterCenters, predictions) = runClustering(data, startingClusterCenters, Array[((Double, Double), Int)](), 0, data)
    var (isEvenlyWeightedRes, clusterSizes, averageClusterSize, stdDev) = isEvenlyWeighted(predictions)
    predictions.show()
    while (!isEvenlyWeightedRes) {
      clusterCenters = splitClusters(predictions, clusterSizes, averageClusterSize, stdDev, clusterCenters)
      val tmp = runClustering(data, clusterCenters, Array[((Double, Double), Int)](), 0, data)
      clusterCenters = tmp._1
      predictions = tmp._2
      val tmp2 = isEvenlyWeighted(predictions)
      isEvenlyWeightedRes = tmp2._1
      clusterSizes = tmp2._2
      averageClusterSize = tmp2._3
      stdDev = tmp2._4
    }

    println(clusterCenters.length)
    clusterCenters.map(_._1)
  }

  private def runClustering(data: Dataset[Row],
    centers: Array[((Double, Double), Int)],
    oldCenters: Array[((Double, Double), Int)],
    iteration: Int,
    clusters: Dataset[Row]): (Array[((Double, Double), Int)], Dataset[Row]) = {

    def nearestClusterFn(x: Double, y: Double): Int = {
      centers.map { r =>
        val dist = Math.sqrt(Math.pow(x - r._1._1, 2) + Math.pow(y - r._1._2, 2))
        (r._2 -> dist)
      }.sortBy(_._2).head._1
    }
    val nearestCluster = udf(nearestClusterFn _)

    if (centers.map { c => // ugh kill me i hate this conditional
      oldCenters.map { o =>
        (o._1._1 -> o._1._2)
      }.contains(((c._1._1 -> c._1._2)))
    }.filter(_ == false).length == 0) {
      (centers, clusters)
    } else {
      println(iteration)
      val clusters = data.withColumn("cluster", nearestCluster(col("x"), col("y")))
      val newCenters = clusters.groupBy("cluster")
        .avg("x", "y")
        .select(col("avg(x)").as("x"), col("avg(y)").as("y"))
        .collect()
        .map { r =>
          (r.getDouble(0) -> r.getDouble(1))
        }.zipWithIndex

      runClustering(data, newCenters, centers, iteration + 1, clusters)
    }
  }

  private def isEvenlyWeighted(predictions: Dataset[Row]): (Boolean, Array[(Int, Int)], Double, Double) = {
    val clusterSizes = predictions
      .groupBy("cluster")
      .count()
      .sort("count")
    val averageClusterSize = clusterSizes
      .select(avg("count"))
      .head()
      .getDouble(0)

    def squareDistFn(x: Double): Double = Math.pow(averageClusterSize - x, 2)
    val squareDist = udf(squareDistFn _)

    val stdDev = clusterSizes
      .withColumn("dist", squareDist(col("count")))
      .select(sum("dist"))
      .head()
      .getDouble(0) / predictions.count()

    (stdDev < stdDevBreakpoint, clusterSizes.collect().map(r => (r.getInt(0) -> r.getLong(1).toInt)), averageClusterSize, stdDev)
  }

  private def splitClusters(predictions: Dataset[Row],
    clusterSizes: Array[(Int, Int)],
    averageClusterSize: Double,
    stdDev: Double,
    clusterCenters: Array[((Double, Double), Int)]): Array[((Double, Double), Int)] = {

    val above = clusterSizes.filter(r => r._2 > (averageClusterSize + stdDev * stdDevTolerance)).sortBy(-_._2)
    val below = clusterSizes.filter(r => r._2 < (averageClusterSize - stdDev * stdDevTolerance)).sortBy(_._2)

    var rover = 0
    for (cluster <- above.map(_._1)) {
      val randomLoc = predictions
        .filter(col("cluster") === cluster.toString())
        .orderBy(rand())
        .take(1)
        .map { r =>
          (r.getDouble(0) -> r.getDouble(1))
        }.head
      clusterCenters(rover) = randomLoc -> clusterCenters(rover)._2
      rover += 1
    }

    clusterCenters
  }
}