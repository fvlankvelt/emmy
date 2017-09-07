package emmy.autodiff


case class UnaryNode[U[_], V, S](up: Node[U, V, S], rf: UnaryValueFunc[V])
                                (implicit
                                 val vt: ValueOps[U, V, S],
                                 val ops: ContainerOps.Aux[U, S])
  extends Node[U, V, S] {

  override val shape = up.shape

  override def apply(ec: EvaluationContext) = {
    ops.map(ec(up))(rf.apply)
  }

  override def grad[W[_], T](gc: GradientContext, v: Variable[W, V, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val opsW = implicitly[ContainerOps[W]]
    val ug = gc(up, v)
    opsW.map(ug) { v =>
      vt.times(v, ops.map(gc(up))(rf.grad))
    }
  }
}
