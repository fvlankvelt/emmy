package emmy.autodiff

import shapeless.Id


trait ConstantLike[U[_], V, S] extends Expression[U, V, S] {

  def value: U[V]

  override def shape = ops.shapeOf(value)

  override def apply(ec: EvaluationContext[V]) = value

  override def grad[W[_], T](gc: GradientContext[V], v: Variable[W, V, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val ops = implicitly[ContainerOps[W]]
    val shape = ops.shapeOf(gc(v))
    ops.fill(shape, vt.zero)
  }
}

case class Constant[U[_], V, S](value: U[V])
                               (implicit
                                val vo: ValueOps[U, V, S],
                                val ops: ContainerOps.Aux[U, S])
  extends ConstantLike[U, V, S] {

  override implicit val vt: ValueOps[U, V, S] = vo.bind(shape)
}

object Constant {

    def apply(value: Double) = Constant[Id, Double, Any](value)
}
