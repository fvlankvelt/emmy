package emmy.inference

import emmy.autodiff.{Constant, Expression, GradientContext, ValueOps, Variable, log, sum}

import scalaz.Scalaz.Id

class AEVBSampler[U[_], V, S](val variable: Variable[U, V, S], val mu: U[V], val sigma: U[V])(implicit idT: ValueOps[Id, V, Any]) {

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
  def update(logP: Expression[Id, V, Any], gc: GradientContext, rho: V): (AEVBSampler[U, V, S], V) = {
    val vt = variable.vt
    val fl = vt.valueVT
    val value = gc(variable)
    implicit val ops = variable.ops
    val gradP = gc(logP, variable)
    val gradQ = gradValue(value)
    val gradDelta = variable.vt.minus(gradP, gradQ)
    val scaledDelta = vt.tanh(
      vt.times(
        variable.ops.fill(variable.shape, rho),
        vt.times(sigma, gradDelta)
      )
    )

    val newMu = vt.plus(
      mu,
      vt.times(
        sigma,
        scaledDelta
      )
    )

    val lambda = vt.log(sigma)
    val newLambda =
      vt.plus(
        lambda,
        vt.times(
          vt.div(
            vt.minus(value, mu),
            vt.times(vt.fromInt(2), sigma)
          ),
          scaledDelta
        )
      )
    val newSigma = vt.exp(newLambda)
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

  def logp(): Expression[Id, V, Any] = {
    implicit val vt = variable.vt
    implicit val numV = vt.valueVT
    implicit val ops = variable.ops
    val x = (variable - Constant(mu)) / Constant(sigma)
    sum(-(log(sigma) + x * x / Constant(vt.fromInt(2))))
  }

  def sample(): U[V] = {
    val vt = variable.vt
    vt.plus(mu, vt.times(vt.rnd, sigma))
  }
}
