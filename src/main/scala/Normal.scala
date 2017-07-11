

case class NormalVector(mu: VectorVariable, sigma: VectorVariable)(implicit model: Model) extends VectorDistribution {

  import Function._
  import Variable._

  private val tau = 1.0f / (sigma * sigma)

  val in: Seq[VariableLike[_]] = Seq(mu, sigma)

  def logp() = {
    val exponent = -tau * (this - mu) ** 2.0f
    val norm: VectorVariable = log(tau / (2 * Math.PI.toFloat))
    sum(exponent + norm) / 2f
  }
}

case class NormalScalar(mu: ScalarVariable, sigma: ScalarVariable)(implicit model: Model) extends ScalarDistribution {

  import Function._
  import Variable._

  private val tau = 1.0f / (sigma * sigma)

  val in: Seq[VariableLike[_]] = Seq(mu, sigma)

  def logp() = {
    (-tau * (this - mu) ** 2.0f + log(tau / (2 * Math.PI.toFloat))) / 2f
  }
}

object Normal {

  def apply(mu: VectorVariable, sigma: VectorVariable)(implicit model: Model) = {
    NormalVector(mu, sigma)
  }

  def apply(mu: ScalarVariable, sigma: ScalarVariable)(implicit model: Model) = {
    NormalScalar(mu, sigma)
  }

}
