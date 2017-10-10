package emmy.autodiff

case class UnaryExpression[U[_], V, S](up: Expression[U, V, S], rf: EvaluableValueFunc[V])
  extends Expression[U, V, S] {

  override val vt = up.vt

  override val ops = up.ops

  override val parents = Seq(up)

  override def apply(ec: EvaluationContext[V]) = {
    val value = ec(up)
    ops.map(value)(v => rf.apply(ec, v))
  }

  override def grad[W[_], T](gc: GradientContext[V], v: Variable[W, V, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val opsW = implicitly[ContainerOps[W]]
    val ug = gc(up, v)
    opsW.map(ug) { g =>
      val v = gc(up)
      vt(gc).times(g, ops.map(v)(u => rf.grad(gc, u)))
    }
  }

  override def toString: String = {
    rf.name + "(" + up + ")"
  }
}
