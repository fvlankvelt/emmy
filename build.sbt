import scalariform.formatter.preferences._

name := "emmy"

version := "0.1"

scalaVersion := "2.12.4"

resolvers += Resolver.bintrayRepo("alexknvl", "maven")

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % "0.13.1",
  "org.scalatest" %% "scalatest" % "3.0.1" % "test",
  "org.scalaz" %% "scalaz-core" % "7.2.14"
)


scalacOptions ++= Seq(
  "-Xlint",
  "-feature",
  "-language:implicitConversions",
  "-language:dynamics",
  "-language:postfixOps",
  "-language:higherKinds",
  "-language:existentials",
  "-language:_",
  "-deprecation",
  "-unchecked"
)

scalariformPreferences := scalariformPreferences.value
  .setPreference(AlignArguments, true)
  .setPreference(AlignParameters, true)
  .setPreference(CompactControlReadability, true)
  .setPreference(RewriteArrowSymbols, true)
  .setPreference(AlignSingleLineCaseStatements, true)
  .setPreference(DoubleIndentConstructorArguments, true)
  .setPreference(DanglingCloseParenthesis, Force)

