package emmy.distribution

import emmy.autodiff.{ConstantLike, ContainerOps, Node, ScalarOps, ValueOps, Variable, log, sum}

import scalaz.Scalaz.Id

case class Normal[U[_], V, S](mu: Node[U, V, S], sigma: Node[U, V, S])
                             (implicit
                              vt: ValueOps[U, V, S],
                              idT: ValueOps[Id, V, Any],
                              ops: ContainerOps.Aux[U, S],
                              so: ScalarOps[V, Double])
  extends Distribution[U, V, S] {

  override def sample =
    NormalSample(mu, sigma)

  override def observe(data: U[V]) =
    NormalObservation(mu, sigma, data)
}

trait NormalStochast[U[_], V, S] extends Stochast[V] {
  self: Node[U, V, S] =>

  def mu: Node[U, V, S]

  def sigma: Node[U, V, S]

  def vt: ValueOps[U, V, S]

  implicit def idT: ValueOps[Id, V, Any]

  implicit def so: ScalarOps[V, Double]

  override def logp(): Node[Id, V, Any] = {
    implicit val numV = vt.valueVT
    val x = (this - mu) / sigma
    sum(-(log(sigma) - x * x / 2.0))
  }
}

case class NormalSample[U[_], V, S](mu: Node[U, V, S], sigma: Node[U, V, S])
                                   (implicit
                                    val vo: ValueOps[U, V, S],
                                    val idT: ValueOps[Id, V, Any],
                                    val ops: ContainerOps.Aux[U, S],
                                    val so: ScalarOps[V, Double])
  extends Variable[U, V, S] with NormalStochast[U, V, S] {

  assert(mu.shape == sigma.shape)

  override val shape = mu.shape

  override implicit val vt = vo.bind(shape)
}

case class NormalObservation[U[_], V, S](mu: Node[U, V, S], sigma: Node[U, V, S], value: U[V])
                                        (implicit
                                         val vo: ValueOps[U, V, S],
                                         val idT: ValueOps[Id, V, Any],
                                         val ops: ContainerOps.Aux[U, S],
                                         val so: ScalarOps[V, Double])
  extends Observation[U, V, S] with NormalStochast[U, V, S] with ConstantLike[U, V, S] {

  assert(mu.shape == sigma.shape)

  override implicit val vt = vo.bind(ops.shapeOf(value))
}
