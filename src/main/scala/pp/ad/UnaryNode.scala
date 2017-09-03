package pp.ad


case class UnaryNode[U[_], V, S](up: Node[U, V, S], rf: UnaryValueFunc[V])(implicit val vt: ValueOps[U, V, S], val ops: ContainerOps.Aux[U, S]) extends Node[U, V, S] {

  override val shape = up.shape

  override def value = ops.map(up())(rf.apply)

  override def calcGrad[W[_], T](v: Variable[W, V, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val opsW = implicitly[ContainerOps[W]]
    val ug = up.grad(v)
    opsW.map(ug) { v =>
      vt.times(v, ops.map(up())(rf.grad))
    }
  }
}
