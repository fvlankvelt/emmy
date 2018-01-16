package emmy.distribution

import emmy.autodiff.ContainerOps.Aux
import emmy.autodiff._

import scala.util.Random
import scalaz.Scalaz
import scalaz.Scalaz.Id

trait CategoricalFactor extends Factor with Node {

  def variable: Expression[Id, Int, Any]

  def thetas: Expression[IndexedSeq, Double, Int]

  override def parents: Seq[Node] = Seq(thetas)

  override val logp: Expression[Id, Double, Any] = {
    val self = this
    new Expression[Id, Double, Any] {

      override val parents = Seq(self)

      override implicit val ops: Aux[Scalaz.Id, Shape] =
        ContainerOps.idOps

      override implicit val so: ScalarOps[Scalaz.Id[Double], Scalaz.Id[Double]] =
        ScalarOps.doubleOps

      override implicit def vt: Evaluable[ValueOps[Scalaz.Id, Double, Any]] =
        ValueOps.idOps(Floating.doubleFloating)

      override def eval(ec: GradientContext): Evaluable[Scalaz.Id[Double]] = {
        val cIndex = ec(variable)
        val cThetas = ec(thetas)
        ctx ⇒ {
          val eIndex = cIndex(ctx)
          val eThetas = cThetas(ctx)
          Floating.doubleFloating.log(eThetas(eIndex) / eThetas.sum)
        }
      }

      override def grad[W[_], T](gc: GradientContext, v: Parameter[W, T]) = {
        val wOps = v.ops
        gc(thetas, v).map { g ⇒
          val cIndex = gc(variable)
          val cThetas = gc(thetas)
          ctx ⇒ {
            val eIndex = cIndex(ctx)
            val eThetas = cThetas(ctx)
            val theta = eThetas(eIndex)
            val thetaSum = eThetas.sum
            wOps.map(g(ctx)) { ug ⇒
              ug(eIndex) / theta - ug.sum / thetaSum
            }
          }
        }
      }

      override def toString = {
        s"logp($self)"
      }
    }
  }
}

trait CategoricalExpression extends Expression[Id, Int, Any] with CategoricalFactor {

  override def vt: Evaluable[ValueOps[Id, Int, Any]] =
    ValueOps(Floating.intFloating, ops, null)

  override val so: ScalarOps[Double, Int] =
    ScalarOps.intDoubleOps
}

class CategoricalSample(val thetas: Expression[IndexedSeq, Double, Int])(implicit val ops: ContainerOps.Aux[Id, Any])
  extends CategoricalVariable with CategoricalExpression {

  override val variable = this

  override val K: Evaluable[Int] = thetas.vt.map(_.shape)

  override def eval(ec: GradientContext): Evaluable[Int] = {
    ec(thetas).map { eThetas ⇒
      val sumThetas = eThetas.sum
      val draw = Random.nextDouble() * sumThetas
      val (_, index) = eThetas.zipWithIndex.foldLeft((draw, 0)) {
        case ((curDraw, curIdx), (theta, idx)) ⇒
          val newDraw = curDraw - theta
          if (curDraw > 0) {
            (newDraw, idx)
          }
          else {
            (newDraw, curIdx)
          }
      }
      index
    }
  }

  override def toString: String = {
    s"~ Cat($thetas)"
  }

}

case class Categorical(thetas: Expression[IndexedSeq, Double, Int])(implicit ops: ContainerOps.Aux[Id, Any])
  extends Distribution[Id, Int, Any] {

  override def sample =
    new CategoricalSample(thetas)

  override def observe(data: Int): Observation[Id, Int, Any] =
    new CategoricalObservation(thetas, data)

  class CategoricalObservation private[Categorical] (
      val thetas: Expression[IndexedSeq, Double, Int],
      val value:  Evaluable[Int]
  )(implicit val ops: ContainerOps.Aux[Id, Any])
    extends Observation[Id, Int, Any] with CategoricalExpression {

    override val variable = this

    override def toString: String = {
      s"<- Cat($thetas)"
    }
  }

}

