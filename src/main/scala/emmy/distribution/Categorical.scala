package emmy.distribution

import emmy.autodiff.ContainerOps.Aux
import emmy.autodiff._

import scala.util.Random
import scalaz.Scalaz
import scalaz.Scalaz.Id

trait CategoricalStochast
  extends Stochast
    with Expression[Id, Int, Any] {

  def thetas: Expression[IndexedSeq, Double, Int]

  override def vt: Evaluable[ValueOps[Id, Int, Any]] =
    ValueOps(Floating.intFloating, ops, null)

  override val so: ScalarOps[Double, Int] = ScalarOps.intDoubleOps

  override def parents = Seq(thetas)

  override def logp(): Expression[Id, Double, Any] = {
    val self = this
    new Expression[Id, Double, Any] {
      override implicit val ops: Aux[Scalaz.Id, Shape] =
        ContainerOps.idOps

      override implicit val so: ScalarOps[Scalaz.Id[Double], Scalaz.Id[Double]] =
        ScalarOps.doubleOps

      override implicit def vt: Evaluable[ValueOps[Scalaz.Id, Double, Any]] =
        ValueOps.idOps(Floating.doubleFloating)

      override def apply(ec: EvaluationContext): Scalaz.Id[Double] = {
        val index = ec(self)
        val thetav = ec(thetas)
        Floating.doubleFloating.log(thetav(index) / thetav.sum)
      }

      override def grad[W[_], T](gc: GradientContext, v: ContinuousVariable[W, T])(implicit wOps: Aux[W, T]): Gradient[W, Scalaz.Id] = {
        val index = gc(self)
        val thetav = gc(thetas)
        val theta = thetav(index)
        val thetaSum = thetav.sum
        val g = gc(thetas, v)
        wOps.map(g) { ug =>
          ug(index) / theta - ug.sum / thetaSum
        }
      }
    }
  }
}

case class Categorical(thetas: Expression[IndexedSeq, Double, Int])
                      (implicit
                       ops: ContainerOps.Aux[Id, Any])
  extends Distribution[Id, Int, Any] {

  override def sample =
    new CategoricalSample(thetas)

  override def observe(data: Int): Observation[Id, Int, Any] =
    new CategoricalObservation(thetas, data)

  class CategoricalSample private[Categorical](val thetas: Expression[IndexedSeq, Double, Int])
                                              (implicit
                                               val ops: ContainerOps.Aux[Id, Any])
    extends IntegerVariable[Id, Any] with CategoricalStochast {

    override def apply(ec: EvaluationContext): Int = {
      val thetasV = ec(thetas)
      val sumThetas = thetasV.sum
      var draw = Random.nextDouble() * sumThetas
      for { (theta, idx) <- thetasV.zipWithIndex } {
        if (theta > draw)
          return idx
        draw -= theta
      }
      throw new UnsupportedOperationException("Uniform draw is larger than 1.0")
    }

    override def toString: String = {
      s"<- Cat($thetas)"
    }
  }


  class CategoricalObservation private[Categorical](val thetas: Expression[IndexedSeq, Double, Int],
                                                    val value: Evaluable[Int])
                                                   (implicit
                                                    val ops: ContainerOps.Aux[Id, Any])
    extends Observation[Id, Int, Any] with CategoricalStochast {

    override def toString: String = {
      s"<- Cat($thetas)"
    }
  }

}

