package pp.ad

case class Scale[U[_], V, S](up: Node[U, V, S], fn: V => V)(implicit val vt: ValueOps[U, V, S], val ops: ContainerOps.Aux[U, S]) extends Node[U, V, S] {

  override val shape = up.shape

  override def value = ops.map(up())(fn)

  override def calcGrad[W[_], T](v: Variable[W, V, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val opsW = implicitly[ContainerOps[W]]
    opsW.map(up.grad(v)) { g => ops.map(g)(fn) }
  }

}
