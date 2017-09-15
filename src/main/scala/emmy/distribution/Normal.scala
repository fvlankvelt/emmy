package emmy.distribution

import emmy.autodiff
import emmy.autodiff.{ContainerOps, EvaluationContext, Expression, Node, ScalarOps, ValueOps, Variable, log, sum}

import scalaz.Scalaz.Id

case class Normal[U[_], V, S](mu: Expression[U, V, S], sigma: Expression[U, V, S])
                             (implicit
                              vt: ValueOps[U, V, S],
                              idT: ValueOps[Id, V, Any],
                              ops: ContainerOps.Aux[U, S],
                              so: ScalarOps[V, Double])
  extends Distribution[U, V, S] {

  override def sample =
    NormalSample(mu, sigma)

  override def observe(data: U[V]): Observation[U, V, S] =
    NormalObservation(mu, sigma, data)
}

trait NormalStochast[U[_], V, S] extends Stochast[V] with Node {
  self: Expression[U, V, S] =>

  def mu: Expression[U, V, S]

  def sigma: Expression[U, V, S]

  def vt: ValueOps[U, V, S]

  implicit def idT: ValueOps[Id, V, Any]

  implicit def so: ScalarOps[V, Double]

  override def parents = Seq(mu, sigma)

  override def logp(): Expression[Id, V, Any] = {
    implicit val numV = vt.valueVT
    val x = (this - mu) / sigma
    sum(-(log(sigma) + x * x / 2.0))
  }
}

case class NormalSample[U[_], V, S](mu: Expression[U, V, S], sigma: Expression[U, V, S])
                                   (implicit
                                    val vo: ValueOps[U, V, S],
                                    val idT: ValueOps[Id, V, Any],
                                    val ops: ContainerOps.Aux[U, S],
                                    val so: ScalarOps[V, Double])
  extends Variable[U, V, S] with NormalStochast[U, V, S] {

  assert(mu.shape == sigma.shape)

  override val shape = mu.shape

  override implicit val vt = vo.bind(shape)

  override def apply(ec: EvaluationContext[V]): U[V] = {
    vt.plus(ec(mu), vt.times(vt.rnd, ec(sigma)))
  }

  override def toString: String = {
    s"~ N($mu, $sigma)"
  }
}

case class NormalObservation[U[_], V, S](mu: Expression[U, V, S], sigma: Expression[U, V, S], value: U[V])
                                        (implicit
                                         val vo: ValueOps[U, V, S],
                                         val idT: ValueOps[Id, V, Any],
                                         val ops: ContainerOps.Aux[U, S],
                                         val so: ScalarOps[V, Double])
  extends Observation[U, V, S] with NormalStochast[U, V, S] {

  assert(mu.shape == sigma.shape)

  override implicit val vt = vo.bind(ops.shapeOf(value))

  override def toString: String = {
    s"<- N($mu, $sigma)"
  }
}
