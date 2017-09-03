package pp.ad

case class Add[U[_], V, S](lhs: Node[U, V, S], rhs: Node[U, V, S])(implicit val vt: ValueOps[U, V, S], val ops: ContainerOps.Aux[U, S]) extends Node[U, V, S] {

  assert(lhs.shape == rhs.shape)

  override val shape = lhs.shape

  override def value = vt.plus(lhs(), rhs())

  override def calcGrad[W[_], T](v: Variable[W, V, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val ops = implicitly[ContainerOps[W]]
    ops.zipMap(lhs.grad(v), rhs.grad(v)) {
      (lg, rg) => vt.plus(lg, rg)
    }
  }

}
