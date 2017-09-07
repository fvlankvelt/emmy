package emmy.autodiff

case class Multiply[U[_], V, S](lhs: Node[U, V, S], rhs: Node[U, V, S])
                               (implicit
                                val vt: ValueOps[U, V, S],
                                val ops: ContainerOps.Aux[U, S])
  extends Node[U, V, S] {

  assert(lhs.shape == rhs.shape)

  override val shape = lhs.shape

  override def apply(ec: EvaluationContext) = {
    vt.times(ec(lhs), ec(rhs))
  }

  override def grad[W[_], T](gc: GradientContext, v: Variable[W, V, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val ops = implicitly[ContainerOps[W]]
    val lv = gc(lhs)
    val leftg = gc(lhs, v)
    val rv = gc(rhs)
    val rightg = gc(lhs, v)
    ops.zipMap(leftg, rightg) {
      (lg, rg) =>
        vt.plus(
          vt.times(lg, rv),
          vt.times(lv, rg)
        )
    }
  }

}
