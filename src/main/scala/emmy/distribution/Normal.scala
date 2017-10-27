package emmy.distribution

import emmy.autodiff._

import scalaz.Scalaz.Id

trait NormalStochast[U[_], S] extends Stochast with Node {
  self: Expression[U, Double, S] =>

  def mu: Expression[U, Double, S]

  def sigma: Expression[U, Double, S]

  override def parents = Seq(mu, sigma)

  override def logp(): Expression[Id, Double, Any] = {
    val x = (this - mu) / sigma
    sum(-(log(sigma) + x * x / 2.0)).toDouble()
  }
}

case class Normal[U[_], S](mu: Expression[U, Double, S],
                              sigma: Expression[U, Double, S])
                             (implicit
                              ops: ContainerOps.Aux[U, S])
  extends Distribution[U, Double, S] {

  override def sample: ContinuousVariable[U, S] =
    new NormalSample(mu, sigma)

  override def observe(data: U[Double]): Observation[U, Double, S] =
    new NormalObservation(mu, sigma, data)

  class NormalSample private[Normal](val mu: Expression[U, Double, S],
                                     val sigma: Expression[U, Double, S])
                                    (implicit
                                     val ops: ContainerOps.Aux[U, S])
    extends ContinuousVariable[U, S] with NormalStochast[U, S] {

    override val vt = mu.vt

    override val so: ScalarOps[U[Double], U[Double]] =
      ScalarOps.liftBoth[U, Double, Double](ScalarOps.doubleOps, ops)

    override def apply(ec: EvaluationContext): U[Double] = {
      val valueT = vt(ec)
      valueT.plus(ec(mu), valueT.times(valueT.rnd, ec(sigma)))
    }

    override def toString: String = {
      s"~ N($mu, $sigma)"
    }
  }

  class NormalObservation private[Normal](val mu: Expression[U, Double, S],
                                          val sigma: Expression[U, Double, S],
                                          val value: Evaluable[U[Double]])
                                         (implicit
                                          val ops: ContainerOps.Aux[U, S])
    extends Observation[U, Double, S] with NormalStochast[U, S] {

    override val vt = mu.vt

    override val so: ScalarOps[U[Double], U[Double]] =
      ScalarOps.liftBoth[U, Double, Double](ScalarOps.doubleOps, ops)

    override def toString: String = {
      s"<- N($mu, $sigma)"
    }
  }

}

