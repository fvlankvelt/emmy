package emmy.distribution

import emmy.autodiff._

import scalaz.Scalaz.Id

trait NormalStochast[U[_], V, S] extends Stochast[V] with Node {
  self: Expression[U, V, S] =>

  def mu: Expression[U, V, S]

  def sigma: Expression[U, V, S]

  def fl: Floating[V]

  def so: ScalarOps[V, Double]

  override def parents = Seq(mu, sigma)

  override def logp(): Expression[Id, V, Any] = {
    implicit val numV = fl
    implicit val scal = so
    val x = (this - mu) / sigma
    sum(-(log(sigma) + x * x / 2.0))
  }
}

case class Normal[U[_], V, S](mu: Expression[U, V, S],
                              sigma: Expression[U, V, S])
                             (implicit
                              fl: Floating[V],
                              ops: ContainerOps.Aux[U, S],
                              so: ScalarOps[V, Double])
  extends Distribution[U, V, S] {

  override def sample =
    new NormalSample(mu, sigma)

  override def observe(data: U[V]): Observation[U, V, S] =
    new NormalObservation(mu, sigma, data)

  class NormalSample private[Normal](val mu: Expression[U, V, S],
                                     val sigma: Expression[U, V, S])
                                    (implicit
                                     val fl: Floating[V],
                                     val ops: ContainerOps.Aux[U, S],
                                     val so: ScalarOps[V, Double])
    extends Variable[U, V, S] with NormalStochast[U, V, S] {

    override val vt = mu.vt

    override def apply(ec: EvaluationContext[V]): U[V] = {
      val valueT = vt(ec)
      valueT.plus(ec(mu), valueT.times(valueT.rnd, ec(sigma)))
    }

    override def toString: String = {
      s"~ N($mu, $sigma)"
    }
  }

  class NormalObservation private[Normal](val mu: Expression[U, V, S],
                                          val sigma: Expression[U, V, S],
                                          val value: Evaluable[U[V]])
                                         (implicit
                                          val fl: Floating[V],
                                          val ops: ContainerOps.Aux[U, S],
                                          val so: ScalarOps[V, Double])
    extends Observation[U, V, S] with NormalStochast[U, V, S] {

    override val vt = mu.vt

    override def toString: String = {
      s"<- N($mu, $sigma)"
    }
  }

}

