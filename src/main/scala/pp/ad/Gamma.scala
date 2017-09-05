package pp.ad

import scalaz.Scalaz.Id

case class Gamma[U[_], V, S](alpha: Node[U, V, S], beta: Node[U, V, S])
                             (implicit
                              vt: ValueOps[U, V, S],
                              idT: ValueOps[Id, V, Any],
                              ops: ContainerOps.Aux[U, S],
                              so: ScalarOps[V, Double])
  extends Distribution[U, V, S] {

  override def observe(data: U[V]) = GammaObservation(alpha, beta, data)
}

trait GammaStochast[U[_], V, S] extends Stochast[V] {
  self: Node[U, V, S] =>

  def alpha: Node[U, V, S]

  def beta: Node[U, V, S]

  def vt: ValueOps[U, V, S]

  implicit def idT: ValueOps[Id, V, Any]

  implicit def so: ScalarOps[V, Double]

  override def logp(): Node[Id, V, Any] = {
    implicit val numV = vt.valueVT
    val regAlpha : Node[U, V, S] = alpha - 1.0
    sum(alpha * log(beta) + regAlpha * log(this) - beta * this - lgamma(alpha))
  }
}

case class GammaObservation[U[_], V, S](mu: Node[U, V, S], sigma: Node[U, V, S], value: U[V])
                                        (implicit
                                         val vo: ValueOps[U, V, S],
                                         val idT: ValueOps[Id, V, Any],
                                         val ops: ContainerOps.Aux[U, S],
                                         val so: ScalarOps[V, Double])
  extends Observation[U, V, S] with NormalStochast[U, V, S] with ConstantLike[U, V, S] {

  assert(mu.shape == sigma.shape)

  override val shape = mu.shape

  override implicit val vt = vo.bind(shape)
}
