package pp

case class Gamma(alpha: ScalarVariableLike, beta: ScalarVariableLike) extends ScalarDistribution {
  import Function._

  override def logp() = alpha * log(beta) + (alpha - 1.0f) * log(this) -beta * this - lgamma(alpha)
}
