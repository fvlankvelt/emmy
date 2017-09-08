package emmy.distribution

import emmy.autodiff.{ConstantLike, ContainerOps, Expression, ScalarOps, ValueOps, Variable, lgamma, log, sum}

import scalaz.Scalaz.Id

case class Gamma[U[_], V, S](alpha: Expression[U, V, S], beta: Expression[U, V, S])
                             (implicit
                              vt: ValueOps[U, V, S],
                              idT: ValueOps[Id, V, Any],
                              ops: ContainerOps.Aux[U, S],
                              so: ScalarOps[V, Double])
  extends Distribution[U, V, S] {

  assert(alpha.shape == beta.shape)

  override def sample =
    GammaSample(alpha, beta)

  override def observe(data: U[V]) =
    GammaObservation(alpha, beta, data)
}

trait GammaStochast[U[_], V, S] extends Stochast[V] {
  self: Expression[U, V, S] =>

  def alpha: Expression[U, V, S]

  def beta: Expression[U, V, S]

  def vt: ValueOps[U, V, S]

  implicit def idT: ValueOps[Id, V, Any]

  implicit def so: ScalarOps[V, Double]

  override def logp(): Expression[Id, V, Any] = {
    implicit val numV = vt.valueVT
    sum(alpha * log(beta) + (alpha - 1.0) * log(this) - beta * this - lgamma(alpha))
  }
}

case class GammaSample[U[_], V, S](alpha: Expression[U, V, S], beta: Expression[U, V, S])
                                   (implicit
                                    vo: ValueOps[U, V, S],
                                    val idT: ValueOps[Id, V, Any],
                                    val ops: ContainerOps.Aux[U, S],
                                    val so: ScalarOps[V, Double])
  extends Variable[U, V, S] with GammaStochast[U, V, S] {

  override val shape = alpha.shape

  override implicit val vt = vo.bind(shape)

  override val parents = Seq(alpha, beta)
}

case class GammaObservation[U[_], V, S](alpha: Expression[U, V, S], beta: Expression[U, V, S], value: U[V])
                                        (implicit
                                         vo: ValueOps[U, V, S],
                                         val idT: ValueOps[Id, V, Any],
                                         val ops: ContainerOps.Aux[U, S],
                                         val so: ScalarOps[V, Double])
  extends Observation[U, V, S] with GammaStochast[U, V, S] {

  assert(alpha.shape == ops.shapeOf(value))

  override implicit val vt = vo.bind(ops.shapeOf(value))

  override val parents = Seq(alpha, beta)
}
