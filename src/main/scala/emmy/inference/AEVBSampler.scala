package emmy.inference

import emmy.autodiff.{Expression, GradientContext, Variable}

import scalaz.Scalaz.Id

class AEVBSampler[U[_], V, S](val variable: Variable[U, V, S], val mu: U[V], sigma: U[V]) {

  /*
  println(s"new sampler for ${variable}: ${mu}, ${sigma}")
  if (mu.asInstanceOf[Double].isNaN ||
    abs(mu.asInstanceOf[Double]) > 20.0 ||
    abs(sigma.asInstanceOf[Double]) > 20.0 ||
    sigma.asInstanceOf[Double].isNaN) {
    assert(false)
  }
  */

  // @formatter:off
  /**
    * Update (\mu, \sigma) by taking a natural gradient step of size \rho.
    * The value is decomposed as \value = \mu + \epsilon * \sigma.
    * Updates are
    *
    *     \mu' = (1 - \rho) * \mu    +
    *              \rho * \sigma**2 * (\gradP - \gradQ)
    *
    *  \sigma' = (1 - \rho) * \sigma +
    *              \rho * (\sigma**2 / 2) * \epsilon * (\gradP - \gradQ)
    *
    * The factors \sigma**2 and \sigma**2/2, respectively, are due to
    * the conversion to natural gradient.  The \epsilon factor is
    * the jacobian d\theta/d\sigma.  (similar factor for \mu is 1)
    */
  // @formatter:on
  def update(logP: Expression[Id, V, Any], gc: GradientContext[V], rho: V): (AEVBSampler[U, V, S], V) = {
    val vt = variable.vt
    val fl = vt.valueVT
    val value = gc(variable)
    implicit val ops = variable.ops
    val gradP = gc(logP, variable)
    val gradQ = gradValue(value)
    val gradDelta = variable.vt.minus(gradQ, gradP)

    val newMu = vt.plus(
      vt.times(
        variable.ops.fill(variable.shape, fl.minus(fl.one, rho)),
        mu
      ),
      vt.times(
        vt.times(sigma, sigma),
        vt.times(variable.ops.fill(variable.shape, rho), gradDelta)
      )
    )

    val lambda = vt.log(sigma)
    val newLambda =
      vt.plus(
        vt.times(
          lambda, variable.ops.fill(variable.shape, fl.minus(fl.one, rho))
        ),
        vt.times(
          vt.div(vt.minus(value, mu), vt.fromInt(2)),
          vt.times(variable.ops.fill(variable.shape, rho), gradDelta)
        )
      )
    val newSigma = vt.exp(vt.div(vt.plus(newLambda, lambda), vt.fromInt(2)))
    val newSampler = new AEVBSampler[U, V, S](variable, newMu, newSigma)
    (newSampler, delta(newSampler))
  }

  def gradValue(value: U[V]): U[V] = {
    val vt = variable.vt
    val delta = vt.minus(mu, value)
    vt.div(delta, vt.times(sigma, sigma))
  }

  def delta(other: AEVBSampler[U, V, S]): V = {
    implicit val vt = variable.vt
    implicit val fl = vt.valueVT
    val ops = variable.ops
    ops.foldLeft(vt.abs(vt.div(vt.minus(mu, other.mu), sigma)))(fl.zero)(fl.sum)
  }

  def sample(): U[V] = {
    val vt = variable.vt
    val value = vt.plus(mu, vt.times(vt.rnd, sigma))
    //      println(s"sampling ${variable}: ${mu}, ${sigma} => $value")
    value
  }
}
