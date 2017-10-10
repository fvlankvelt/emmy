package emmy.distribution

import emmy.autodiff.{ContainerOps, Evaluable, EvaluationContext, Expression, Floating, ScalarOps, Variable, lgamma, log, sum}

import scalaz.Scalaz.Id

trait GammaStochast[U[_], V, S] extends Stochast[V] {
  self: Expression[U, V, S] =>

  def alpha: Expression[U, V, S]

  def beta: Expression[U, V, S]

  def fl: Floating[V]

  def so: ScalarOps[V, Double]

  override def parents = Seq(alpha, beta)

  override def logp(): Expression[Id, V, Any] = {
    implicit val numV = fl
    implicit val iso = so
    sum(alpha * log(beta) + (alpha - 1.0) * log(this) - beta * this - lgamma(alpha))
  }
}

case class Gamma[U[_], V, S](alpha: Expression[U, V, S], beta: Expression[U, V, S])
                            (implicit
                             fl: Floating[V],
                             ops: ContainerOps.Aux[U, S],
                             so: ScalarOps[V, Double])
  extends Distribution[U, V, S] {

  //  assert(alpha.shape == beta.shape)

  override def sample =
    new GammaSample(alpha, beta)

  override def observe(data: U[V]) =
    new GammaObservation(alpha, beta, data)

  class GammaSample private[Gamma](val alpha: Expression[U, V, S],
                                   val beta: Expression[U, V, S])
                                  (implicit
                                   val fl: Floating[V],
                                   val ops: ContainerOps.Aux[U, S],
                                   val so: ScalarOps[V, Double])
    extends Variable[U, V, S] with GammaStochast[U, V, S] {

    override val vt = alpha.vt

    override def apply(ec: EvaluationContext[V]): U[V] = {
      val alphaV = ec(alpha)
      val betaV = ec(beta)
      ops.zipMap(alphaV, betaV) {
        (a, b) =>
          val valueVT = vt(ec).valueVT
          val g = breeze.stats.distributions.Gamma(valueVT.toDouble(a), 1.0 / valueVT.toDouble(b))
          so.times(valueVT.one, g.draw())
      }
    }
  }

  class GammaObservation private[Gamma](val alpha: Expression[U, V, S],
                                        val beta: Expression[U, V, S],
                                        val value: Evaluable[U[V]])
                                       (implicit
                                        val fl: Floating[V],
                                        val ops: ContainerOps.Aux[U, S],
                                        val so: ScalarOps[V, Double])
    extends Observation[U, V, S] with GammaStochast[U, V, S] {

    override val vt = alpha.vt
  }

}

