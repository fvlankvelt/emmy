name := "emmy"

version := "0.1"

scalaVersion := "2.11.11"

resolvers += Resolver.bintrayRepo("alexknvl", "maven")

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % "0.13.1",
  "org.scalatest" %% "scalatest" % "3.0.1" % "test",
  "org.scalaz" %% "scalaz-core" % "7.2.14"
)


