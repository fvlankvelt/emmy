package pp.ad

case class Multiply[U[_], V, S](lhs: Node[U, V, S], rhs: Node[U, V, S])(implicit val vt: ValueOps[U, V], val ops: ContainerOps.Aux[U, S]) extends Node[U, V, S] {

  assert(lhs.shape == rhs.shape)

  override val shape = lhs.shape

  override def value = vt.times(lhs(), rhs())

  override def calcGrad[W[_], T](v: Variable[W, V, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val ops = implicitly[ContainerOps[W]]
    val lv = lhs()
    val leftg = lhs.grad(v)
    val rv = rhs()
    val rightg = rhs.grad(v)
    ops.zipMap(leftg, rightg) {
      (lg, rg) =>
        vt.plus(
          vt.times(lg, rv),
          vt.times(lv, rg)
        )
    }
  }

}
