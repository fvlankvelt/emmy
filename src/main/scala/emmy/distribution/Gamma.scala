package emmy.distribution

import emmy.autodiff.{ ContainerOps, ContinuousVariable, Evaluable, EvaluationContext, Expression, ScalarOps, ValueOps, lgamma, log, sum }

import scalaz.Scalaz.Id

trait GammaFactor[U[_], S] extends Factor {
  self: Expression[U, Double, S] ⇒

  def alpha: Expression[U, Double, S]

  def beta: Expression[U, Double, S]

  override def parents = Seq(alpha, beta)

  override val logp: Expression[Id, Double, Any] = {
    sum(alpha * log(beta) + (alpha - 1.0) * log(this) - beta * this - lgamma(alpha)).toDouble
  }
}

case class Gamma[U[_], S](alpha: Expression[U, Double, S], beta: Expression[U, Double, S])(implicit ops: ContainerOps.Aux[U, S])
  extends Distribution[U, Double, S] {

  //  assert(alpha.shape == beta.shape)

  override def sample =
    new GammaSample(alpha, beta)

  override def observe(data: U[Double]) =
    new GammaObservation(alpha, beta, data)

  class GammaSample private[Gamma] (
      val alpha: Expression[U, Double, S],
      val beta:  Expression[U, Double, S]
  )(implicit val ops: ContainerOps.Aux[U, S])
    extends ContinuousVariable[U, S] with GammaFactor[U, S] {

    override val vt: Evaluable[ValueOps[U, Double, S]] =
      alpha.vt

    override val so: ScalarOps[U[Double], U[Double]] =
      ScalarOps.liftBoth[U, Double, Double](ScalarOps.doubleOps, ops)

    override def apply(ec: EvaluationContext): U[Double] = {
      val alphaV = ec(alpha)
      val betaV = ec(beta)
      ops.zipMap(alphaV, betaV) {
        (a, b) ⇒
          val valueVT = vt(ec).valueVT
          val g = breeze.stats.distributions.Gamma(valueVT.toDouble(a), 1.0 / valueVT.toDouble(b))
          g.draw()
      }
    }
  }

  class GammaObservation private[Gamma] (
      val alpha: Expression[U, Double, S],
      val beta:  Expression[U, Double, S],
      val value: Evaluable[U[Double]]
  )(implicit val ops: ContainerOps.Aux[U, S])
    extends Observation[U, Double, S] with GammaFactor[U, S] {

    override val vt: Evaluable[ValueOps[U, Double, S]] =
      alpha.vt

    override val so: ScalarOps[U[Double], U[Double]] =
      ScalarOps.liftBoth[U, Double, Double](ScalarOps.doubleOps, ops)
  }

}

