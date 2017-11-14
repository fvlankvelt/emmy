package emmy.inference.aevb

import emmy.autodiff.{ Constant, ContinuousVariable, Evaluable, EvaluationContext, Expression, GradientContext, ValueOps, Visitor, log, sum }
import emmy.inference.Sampler

import scalaz.Scalaz.Id

class ContinuousSampler[U[_], S](
    val variable: ContinuousVariable[U, S],
    val mu:       U[Double],
    val sigma:    U[Double]
) extends Sampler {

  // @formatter:off
  override val parents = Seq(variable)

  override def visit[R](visitor: Visitor[R]) = visitor.visitSampler(this)

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
   *
   * Updates are taken through a tanh to prevent updates from taking the
   * mean too far from it's current value.  The scale is set by the stddev.
   */
  // @formatter:on
  def update(logP: Seq[Expression[Id, Double, Any]], gc: GradientContext, rho: Double): (ContinuousSampler[U, S], Double) = {
    val vt = variable.vt(gc)
    val value = gc(variable)
    implicit val ops = variable.ops
    val gradients = logP.flatMap {
      gc(_, variable)
    }
    if (gradients.isEmpty) {
      return (this, 0.0)
    }
    val gradP = gradients.reduce(vt.plus)
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
    val newSampler = new ContinuousSampler[U, S](variable, newMu, newSigma)
    (newSampler, delta(newSampler, vt))
  }

  def gradValue(value: U[Double], vt: ValueOps[U, Double, S]): U[Double] = {
    val delta = vt.minus(mu, value)
    vt.div(delta, vt.times(sigma, sigma))
  }

  def delta(other: ContinuousSampler[U, S], vt: ValueOps[U, Double, S]): Double = {
    val ops = variable.ops
    ops.foldLeft(vt.abs(vt.div(vt.minus(mu, other.mu), sigma)))(0.0)(_ + _)
  }

  override val logp: Expression[Id, Double, Any] = {
    implicit val vt = variable.vt
    implicit val ops = variable.ops
    val x = (variable - Constant(mu)) / Constant(sigma)
    sum(-(log(Constant(Evaluable.fromConstant(sigma))) + x * x / Constant(vt.map(_.fromInt(2)))))
  }

  def sample(ec: EvaluationContext): U[Double] = {
    val vt = variable.vt(ec)
    vt.plus(mu, vt.times(vt.rnd, sigma))
  }
}
