lazy val root = (project in file("."))
    .settings(
        name                := "SimulatingAntMovement",
        organization        := "edu.trinity",
        scalaVersion        := "2.11.8",
        version             := "0.1.0-SNAPSHOT",
        libraryDependencies += "org.scalafx" % "scalafx_2.11" % "8.0.102-R11",
        //libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "2.2.0" % "provided",
        //libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.2.0" % "provided"
        libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "2.2.0",
        libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.2.0"
)

assemblyMergeStrategy in assembly := {
 case PathList("META-INF", xs @ _*) => MergeStrategy.discard
 case x => MergeStrategy.first
}
