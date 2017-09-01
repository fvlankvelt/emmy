package pp.ad

import scalaz.Scalaz.Id

case class Normal[U[_], V, S](mu: Node[U, V, S], sigma: Node[U, V, S])
                             (implicit
                              vt: ValueOps[U, V],
                              idT: ValueOps[Id, V],
                              ops: ContainerOps.Aux[U, S],
                              so: ScalarOps[V, Double])
  extends Distribution[U, V, S] {

  override def sample(implicit model: Model) = NormalSample(mu, sigma)

  override def observe(data: U[V]) = NormalObservation(mu, sigma, data)
}

trait NormalStochast[U[_], V, S] extends Stochast[V] {
  self: Node[U, V, S] =>

  def mu: Node[U, V, S]

  def sigma: Node[U, V, S]

  def vt: ValueOps[U, V]

  implicit def idT: ValueOps[Id, V]

  implicit def so: ScalarOps[V, Double]

  override def logp(): Node[Id, V, Any] = {
    implicit val numV = vt.valueVT
    val x = (this - mu) / sigma
    sum(-(log(sigma) - x * x / 2.0))
  }
}

case class NormalSample[U[_], V, S](mu: Node[U, V, S], sigma: Node[U, V, S])
                                   (implicit
                                    val vt: ValueOps[U, V],
                                    val idT: ValueOps[Id, V],
                                    val ops: ContainerOps.Aux[U, S],
                                    val so: ScalarOps[V, Double],
                                    model: Model)
  extends Sample[U, V, S] with NormalStochast[U, V, S] {

  assert(mu.shape == sigma.shape)

  override val shape = mu.shape

  override def value = model.valueOf(this)
}

case class NormalObservation[U[_], V, S](mu: Node[U, V, S], sigma: Node[U, V, S], value: U[V])
                                        (implicit
                                         val vt: ValueOps[U, V],
                                         val idT: ValueOps[Id, V],
                                         val ops: ContainerOps.Aux[U, S],
                                         val so: ScalarOps[V, Double])
  extends Observation[U, V, S] with NormalStochast[U, V, S] with ConstantLike[U, V, S] {

  assert(mu.shape == sigma.shape)

  override val shape = mu.shape
}
