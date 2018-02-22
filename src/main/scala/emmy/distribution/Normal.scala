package emmy.distribution

import emmy.autodiff._

import scalaz.Scalaz.Id

trait NormalFactor[U[_], S] extends Factor with Node {

  def mu: Expression[U, Double, S]

  def sigma: Expression[U, Double, S]

  def variable: Expression[U, Double, S]

  override def parents = Seq(mu, sigma)

  override lazy val logp: Expression[Id, Double, Any] = {
    implicit val ops = variable.ops
    val x = (variable - mu) / sigma
    sum(-(log(sigma) + x * x / 2.0)).toDouble
  }
}

class UnitNormalSample[U[_], S](implicit
    val ops: ContainerOps.Aux[U, S],
                                val so: ScalarOps[U[Double], U[Double]],
                                val vt: Evaluable[ValueOps[U, Double, S]]
)
  extends Expression[U, Double, S] {

  override def eval(ec: GradientContext): Evaluable[U[Double]] = {
    ctx ⇒
      {
        val valueT = vt(ctx)
        valueT.rnd
      }
  }
}

class NormalSample[U[_], S](
    val mu:    Expression[U, Double, S],
    val sigma: Expression[U, Double, S]
)(implicit val ops: ContainerOps.Aux[U, S])
  extends ContinuousVariable[U, S] with NormalFactor[U, S] {

  override val variable: NormalSample[U, S] = this

  override val vt: Evaluable[ValueOps[U, Double, S]] =
    mu.vt

  override val so: ScalarOps[U[Double], U[Double]] =
    ScalarOps.liftBoth[U, Double, Double](ScalarOps.doubleOps, ops)

  private val unit = new UnitNormalSample()(ops, so, vt)
  private val upstream = mu + unit * sigma

  override def eval(ec: GradientContext): Evaluable[U[Double]] = {
    ec(upstream)
  }

  override def grad[W[_], T](gc: GradientContext, v: Parameter[W, T]): Gradient[W, U] = {
    gc(upstream, v)
  }

  override def toString: String = {
    s"~ N($mu, $sigma)"
  }
}

case class Normal[U[_], S](
    mu:    Expression[U, Double, S],
    sigma: Expression[U, Double, S]
)(implicit ops: ContainerOps.Aux[U, S])
  extends Distribution[U, Double, S] {

  override def sample: ContinuousVariable[U, S] =
    new NormalSample(mu, sigma)

  def factor(v: Expression[U, Double, S]) = {
    val self = this
    new NormalFactor[U, S] {

      override val mu: Expression[U, Double, S] = self.mu

      override val sigma: Expression[U, Double, S] = self.sigma

      override val variable: Expression[U, Double, S] = v
    }
  }

  override def observe(data: U[Double]): Observation[U, Double, S] =
    new NormalObservation(mu, sigma, data)

  class NormalObservation private[Normal] (
      val mu:    Expression[U, Double, S],
      val sigma: Expression[U, Double, S],
      val value: Evaluable[U[Double]]
  )(implicit val ops: ContainerOps.Aux[U, S])
    extends Observation[U, Double, S] with NormalFactor[U, S] {

    override val variable: NormalObservation = this

    override val vt: Evaluable[ValueOps[U, Double, S]] =
      mu.vt

    override val so: ScalarOps[U[Double], U[Double]] =
      ScalarOps.liftBoth[U, Double, Double](ScalarOps.doubleOps, ops)

    override def toString: String = {
      s"($value <- N($mu, $sigma))"
    }
  }

}

