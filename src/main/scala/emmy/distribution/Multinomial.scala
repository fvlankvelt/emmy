package emmy.distribution

import emmy.autodiff._

import scalaz.Scalaz.Id

trait MultinomialStochast[U[_], S] extends Stochast with Node {
  self: Expression[U, Int, S] =>

  def thetas: Expression[U, Double, S]

  def n: Expression[Id, Int, Any]

  override def parents = Seq(thetas)

  override def logp(): Expression[Id, Double, Any] = {
    lgamma((n + 1).toDouble()) - sum(lgamma((this + 1).toDouble())) + sum(this.toDouble() * log(thetas))
  }
}

case class Multinomial[U[_], S](thetas: Expression[U, Double, S],
                                n: Expression[Id, Int, Any])
                               (implicit
                                ops: ContainerOps.Aux[U, S])
  extends Distribution[U, Int, S] {

  override def sample =
    throw new UnsupportedOperationException("Discrete variables are not supported")

  override def observe(data: U[Int]): Observation[U, Int, S] =
    new MultinomialObservation(thetas, n, data)

  class MultinomialObservation private[Multinomial](val thetas: Expression[U, Double, S],
                                                    val n: Expression[Id, Int, Any],
                                                    val value: Evaluable[U[Int]])
                                                   (implicit
                                                    val ops: ContainerOps.Aux[U, S])
    extends Observation[U, Int, S] with MultinomialStochast[U, S] {

    override def vt: Evaluable[ValueOps[U, Int, S]] =
      thetas.vt.map(v => ValueOps(Floating.intFloating, ops, v.shape))

    override val so: ScalarOps[U[Double], U[Int]] =
      ScalarOps.liftBoth[U, Double, Int](ScalarOps.intDoubleOps, ops)

    override def toString: String = {
      s"<- Multi($thetas, $n)"
    }
  }

}

