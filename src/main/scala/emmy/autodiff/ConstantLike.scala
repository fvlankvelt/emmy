package emmy.autodiff

import scalaz.Scalaz.Id

trait ConstantLike[U[_], V, S] extends Expression[U, V, S] {

  def value: Evaluable[U[V]]

  override def apply(ec: EvaluationContext) = value(ec)

  override def grad[W[_], T](gc: GradientContext, v: Variable[W, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val ops = implicitly[ContainerOps[W]]
    val shape = ops.shapeOf(gc(v))
    ops.fill(shape, vt(gc).forDouble.zero)
  }

  override def toString: String = {
    value.toString
  }
}

case class Constant[U[_], V, S](value: Evaluable[U[V]])
                               (implicit
                                val fl: Floating[V],
                                val so: ScalarOps[U[Double], U[V]],
                                val ops: ContainerOps.Aux[U, S])
  extends ConstantLike[U, V, S] {

  override val vt = value.map(toVT)

  private def toVT(v: U[V]) = {
    val shape = ops.shapeOf(v)
    ValueOps(fl, ops, shape)
  }
}

object Constant {

  def apply[U[_], V, S](value: U[V])
                       (implicit
                        fl: Floating[V],
                        so: ScalarOps[U[Double], U[V]],
                        ops: ContainerOps.Aux[U, S]): Constant[U, V, S] =
    Constant(Evaluable.fromConstant(value))

  def apply(value: Double): Constant[Id, Double, Any] = {
    Constant[Id, Double, Any](value)(Floating.doubleFloating, ScalarOps.doubleOps, ContainerOps.idOps)
  }
}
