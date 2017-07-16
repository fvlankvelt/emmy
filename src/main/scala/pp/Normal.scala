package pp


case class NormalVector(mu: VectorVariableLike, sigma: VectorVariableLike) extends VectorDistribution {

  assert(mu.length == sigma.length)

  val length = mu.length

  import Function._
  import Variable._

  private val tau : VectorVariableLike = 1.0f / (sigma * sigma)

  def logp() = {
    val exponent = -tau * (this - mu) ** 2.0f
    val norm: VectorVariableLike = log(tau / (2 * Math.PI.toFloat))
    sum(exponent + norm) / 2f
  }

}

case class NormalScalar private(mu: ScalarVariableLike, sigma: ScalarVariableLike) extends ScalarDistribution {

  import Function._
  import Variable._

  private val tau = 1.0f / (sigma * sigma)

  def logp() = {
    (-tau * (this - mu) ** 2.0f + log(tau / (2 * Math.PI.toFloat))) / 2f
  }

}

object Normal {

  def apply(mu: VectorVariableLike, sigma: VectorVariableLike) = {
    NormalVector(mu, sigma)
  }

  def apply(mu: ScalarVariableLike, sigma: ScalarVariableLike) = {
    NormalScalar(mu, sigma)
  }

}