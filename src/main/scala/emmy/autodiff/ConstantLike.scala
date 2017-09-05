package emmy.autodiff


trait ConstantLike[U[_], V, S] extends Node[U, V, S] {

  override def calcGrad[W[_], T](v: Variable[W, V, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val ops = implicitly[ContainerOps[W]]
    val shape = ops.shapeOf(v())
    ops.fill(shape, vt.zero)
  }
}

case class Constant[U[_], V, S](value: U[V])(implicit val vo: ValueOps[U, V, S], val ops: ContainerOps.Aux[U, S]) extends ConstantLike[U, V, S] {

  override val shape = ops.shapeOf(value)

  override implicit val vt : ValueOps[U, V, S] = vo.bind(shape)
}

