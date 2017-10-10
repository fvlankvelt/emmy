package emmy.inference

import emmy.autodiff.{Constant, Evaluable, EvaluationContext, Expression, Floating, GradientContext, ValueOps, Variable, log, sum}

import scalaz.Scalaz.Id

class AEVBSampler[U[_], V, S](val variable: Variable[U, V, S],
                              val mu: U[V],
                              val sigma: U[V])
                             (implicit
                              fl: Floating[V]) {

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
    val vt = variable.vt(gc)
    val fl = vt.valueVT
    val value = gc(variable)
    implicit val ops = variable.ops
    val gradP = gc(logP, variable)
    val gradQ = gradValue(value, vt)
    val gradDelta = vt.minus(gradP, gradQ)
    val scaledDelta = vt.tanh(
      vt.times(
        variable.ops.fill(vt.shape, rho),
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
    val newSampler = new AEVBSampler[U, V, S](variable, newMu, newSigma)(fl)
    (newSampler, delta(newSampler, vt))
  }

  def gradValue(value: U[V], vt: ValueOps[U, V, S]): U[V] = {
    val delta = vt.minus(mu, value)
    vt.div(delta, vt.times(sigma, sigma))
  }

  def delta(other: AEVBSampler[U, V, S], vt: ValueOps[U, V, S]): V = {
    implicit val fl = vt.valueVT
    val ops = variable.ops
    ops.foldLeft(vt.abs(vt.div(vt.minus(mu, other.mu), sigma)))(fl.zero)(fl.sum)
  }

  def logp(): Expression[Id, V, Any] = {
    implicit val vt = variable.vt
    implicit val ops = variable.ops
    val x = (variable - Constant(mu)) / Constant(sigma)
    sum(-(log(Constant(Evaluable.fromConstant(sigma))) + x * x / Constant(vt.map(_.fromInt(2)))))
  }

  def sample(ec: EvaluationContext[V]): U[V] = {
    val vt = variable.vt(ec)
    vt.plus(mu, vt.times(vt.rnd, sigma))
  }
}
