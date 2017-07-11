trait Op {
  val in: Seq[VariableLike[_]]
}

trait Distribution extends Op {
  def logp(): ScalarVariableLike
}

trait VectorDistribution extends VectorVariableLike with Distribution

trait ScalarDistribution extends ScalarVariableLike with Distribution
