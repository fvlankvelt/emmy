package emmy.inference.aevb

import emmy.autodiff._
import emmy.distribution.{ Normal, NormalSample }
import emmy.inference.{ Sampler, SamplerBuilder }

import scalaz.Scalaz
import scalaz.Scalaz.Id
//
//class ContinuousSampler[U[_], S](
//                                  val variable: ContinuousVariable[U, S],
//                                  val cVal: Evaluable[U[Double]],
//                                  val grad: Evaluable[U[Double]],
//                                  var mu: U[Double],
//                                  var sigma: U[Double]
//                                ) extends Sampler {
//
//  override val parents = Seq(variable)
//
//  override def visit[R](visitor: Visitor[R]): R =
//    visitor.visitSampler(this)
//
//  val Q = new NormalSample(
//    Constant(((_: SampleContext) => mu) : Evaluable[U[Double]])(Floating.doubleFloating, variable.so, variable.ops),
//    Constant(((_: SampleContext) => sigma) : Evaluable[U[Double]])(Floating.doubleFloating, variable.so, variable.ops)
//  )(variable.ops)
//
//  // @formatter:off
//  /**
//   * Update (\mu, \sigma) by taking a natural gradient step of size \rho.
//   * The value is decomposed as \value = \mu + \epsilon * \sigma.
//   * Updates are
//   *
//   *     \mu' = (1 - \rho) * \mu    +
//   *              \rho * \sigma**2 * (\gradP - \gradQ)
//   *
//   *  \sigma' = (1 - \rho) * \sigma +
//   *              \rho * (\sigma**2 / 2) * \epsilon * (\gradP - \gradQ)
//   *
//   * The factors \sigma**2 and \sigma**2/2, respectively, are due to
//   * the conversion to natural gradient.  The \epsilon factor is
//   * the jacobian d\theta/d\sigma.  (similar factor for \mu is 1)
//   *
//   * Updates are taken through a tanh to prevent updates from taking the
//   * mean too far from it's current value.  The scale is set by the stddev.
//   */
//  // @formatter:on
//  def update(sc: SampleContext, rho: Double): Double = {
//    val vt = variable.vt(sc)
//    val value = cVal(sc)
//    implicit val ops = variable.ops
//    val gradP = grad(sc)
//    val gradQ = gradValue(value, vt)
//    val gradDelta = vt.minus(gradP, gradQ)
//    val scaledDelta = vt.tanh(
//      vt.times(
//        variable.ops.fill(vt.shape, rho),
//        vt.times(sigma, gradDelta)
//      )
//    )
//
//    val newMu = vt.plus(
//      mu,
//      vt.times(
//        sigma,
//        scaledDelta
//      )
//    )
//
//    val lambda = vt.log(sigma)
//    val newLambda =
//      vt.plus(
//        lambda,
//        vt.times(
//          vt.div(
//            vt.minus(value, mu),
//            vt.times(vt.fromInt(2), sigma)
//          ),
//          scaledDelta
//        )
//      )
//    val newSigma = vt.exp(newLambda)
//    val d = delta(newMu, newSigma, vt)
//    mu = newMu
//    sigma = newSigma
//    d
//  }
//
//  def gradValue(value: U[Double], vt: ValueOps[U, Double, S]): U[Double] = {
//    val delta = vt.minus(mu, value)
//    vt.div(delta, vt.times(sigma, sigma))
//  }
//
//  def delta(newMu: U[Double], newSigma: U[Double], vt: ValueOps[U, Double, S]): Double = {
//    val ops = variable.ops
//    ops.foldLeft(vt.abs(vt.div(vt.minus(mu, newMu), sigma)))(0.0)(_ + _)
//  }
//
//  override val logp: Expression[Id, Double, Any] = {
//    implicit val vt = variable.vt
//    implicit val ops = variable.ops
//    val x = (variable - Constant(mu)) / Constant(sigma)
//    sum(-(log(Constant(Evaluable.fromConstant(sigma))) + x * x / Constant(vt.map(_.fromInt(2)))))
//  }
//
//  def sample(ec: SampleContext): U[Double] = {
//    val vt = variable.vt(ec)
//    vt.plus(mu, vt.times(vt.rnd, sigma))
//  }
//
//  override def toBuilder(ec: GradientContext): SamplerBuilder = {
//    val self = this
//    new SamplerBuilder {
//
//      override def variable: Node = self.variable
//
//      override def build(logP: Seq[Expression[Scalaz.Id, Double, Any]]): Sampler = {
//        implicit val ops = self.variable.ops
//        val prior = Normal(Constant(mu), Constant(sigma)).sample
//        new ContinuousSampler[U, S](self.variable, prior.eval(ec), logp.grad(ec, self.variable).get, mu, sigma)
//      }
//
//    }
//  }
//}
