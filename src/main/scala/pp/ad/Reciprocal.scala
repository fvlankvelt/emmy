package pp.ad

case class Reciprocal[U[_], V, S](upstream: Node[U, V, S])(implicit val vt: ValueOps[U, V, S], val ops: ContainerOps.Aux[U, S]) extends Node[U, V, S] {

  override val shape = upstream.shape

  override def value = vt.div(vt.one, upstream())

  override def calcGrad[W[_], T](v: Variable[W, V, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val ops = implicitly[ContainerOps[W]]
    val value = upstream()
    val grad = upstream.grad(v)
    ops.map(grad) { g =>
      vt.div(g, vt.times(value, value))
    }
  }

}
