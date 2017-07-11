

case class NormalVector(mu: VectorVariableLike, sigma: VectorVariableLike)(implicit model: Model) extends VectorDistribution {

  assert(mu.length == sigma.length)

  val length = mu.length

  import Function._
  import Variable._

  private val tau : VectorVariableLike = 1.0f / (sigma * sigma)

  val in: Seq[VariableLike[_]] = Seq(mu, sigma)

  def logp() = {
    val exponent = -tau * (this - mu) ** 2.0f
    val norm: VectorVariableLike = log(tau / (2 * Math.PI.toFloat))
    sum(exponent + norm) / 2f
  }
}

case class NormalScalar(mu: ScalarVariableLike, sigma: ScalarVariableLike)(implicit model: Model) extends ScalarDistribution {

  import Function._
  import Variable._

  private val tau = 1.0f / (sigma * sigma)

  val in: Seq[VariableLike[_]] = Seq(mu, sigma)

  def logp() = {
    (-tau * (this - mu) ** 2.0f + log(tau / (2 * Math.PI.toFloat))) / 2f
  }
}

object Normal {

  def apply(mu: VectorVariableLike, sigma: VectorVariableLike)(implicit model: Model) = {
    NormalVector(mu, sigma)
  }

  def apply(mu: ScalarVariableLike, sigma: ScalarVariableLike)(implicit model: Model) = {
    NormalScalar(mu, sigma)
  }

}
