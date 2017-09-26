package emmy.autodiff


case class UnaryExpression[U[_], V, S](up: Expression[U, V, S], rf: UnaryValueFunc[V])
  extends Expression[U, V, S] {

  override val vt = up.vt

  override val ops = up.ops

  override val shape = up.shape

  override val parents = Seq(up)

  override def apply(ec: EvaluationContext) = {
    ops.map(ec(up))(rf.apply)
  }

  override def grad[W[_], T](gc: GradientContext, v: Variable[W, V, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val opsW = implicitly[ContainerOps[W]]
    val ug = gc(up, v)
    opsW.map(ug) { g =>
      val v = gc(up)
      vt.times(g, ops.map(v)(rf.grad))
    }
  }

  override def toString: String = {
    rf.name + "(" + up + ")"
  }
}
