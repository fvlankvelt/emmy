trait Op {
  val in: Seq[VariableLike[_]]
}

trait Distribution extends Op {
  def logp(): VariableLike[Float]
}

trait VectorDistribution extends VectorVariable with Distribution

trait ScalarDistribution extends ScalarVariable with Distribution
