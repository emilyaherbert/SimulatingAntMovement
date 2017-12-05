package misc

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.IntegerType

/**
 * Clustering on the BRFSS data set. https://www.cdc.gov/brfss/
 */
object KMeansClustering {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.appName("Simple Application").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val csvData = spark.read.option("header", true).csv("/users/mlewis/CSCI3395-F17/data/LLCP2015.csv")

    val columnsToKeep = "GENHLTH PHYSHLTH MENTHLTH POORHLTH EDUCA SEX MARITAL EMPLOY1".split(" ")

    val typedData = columnsToKeep.foldLeft(csvData)((df, colName) => df.withColumn(colName, df(colName).cast(IntegerType).as(colName))).na.drop()
    val assembler = new VectorAssembler().setInputCols(columnsToKeep).setOutputCol("features")
    val dataWithFeatures = assembler.transform(typedData)
    dataWithFeatures.show()

    val normalizer = new Normalizer().setInputCol("features").setOutputCol("normFeatures")
    val normData = normalizer.transform(dataWithFeatures)

    val kmeans = new KMeans().setK(5).setFeaturesCol("normFeatures")
    val model = kmeans.fit(normData)

    val cost = model.computeCost(normData)
    println("total cost = " + cost)
    println("cost distance = " + math.sqrt(cost / normData.count()))

    val predictions = model.transform(normData)
    predictions.select("features", "prediction").show()
    
    model.clusterCenters.foreach(println)

    spark.stop()
  }
}