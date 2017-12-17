lazy val root = (project in file("."))
    .settings(
        name                := "SimulatingAntMovement",
        organization        := "edu.trinity",
        scalaVersion        := "2.11.8",
        version             := "0.1.0-SNAPSHOT",
        libraryDependencies += "org.scalafx" % "scalafx_2.11" % "8.0.102-R11",
        //libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "2.2.0" % "provided",
        //libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.2.0" % "provided",
        //libraryDependencies += "org.apache.spark" % "spark-sql_2.11" % "2.2.0" % "provided",
        libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "2.2.0",
        libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.2.0",
        libraryDependencies += "org.apache.spark" % "spark-sql_2.11" % "2.2.0",
        libraryDependencies += "com.twelvemonkeys.imageio" % "imageio-core" % "3.3.2",
        libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "0.9.1",
        libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.9.1",
        libraryDependencies += "org.deeplearning4j" % "dl4j-spark_2.11" % "0.9.1_spark_2"
)

assemblyMergeStrategy in assembly := {
 case PathList("META-INF", xs @ _*) => MergeStrategy.discard
 case x => MergeStrategy.first
}
