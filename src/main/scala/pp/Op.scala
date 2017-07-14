package pp

trait Distribution {
  def logp(): ScalarVariableLike
}

trait VectorDistribution extends VectorVariableLike with Distribution

trait ScalarDistribution extends ScalarVariableLike with Distribution
