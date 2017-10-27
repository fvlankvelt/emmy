package emmy.autodiff

case class UnaryExpression[U[_], V, S](up: Expression[U, V, S], rf: EvaluableValueFunc[V])
  extends Expression[U, V, S] {

  override val vt = up.vt

  override val ops = up.ops

  override val so = up.so

  override val parents = Seq(up)

  override def apply(ec: EvaluationContext) = {
    val value = ec(up)
    ops.map(value)(v => rf.apply(ec, v))
  }

  override def grad[W[_], T](gc: GradientContext, v: ContinuousVariable[W, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val opsW = implicitly[ContainerOps[W]]
    val ug = gc(up, v)
    opsW.map(ug) { g =>
      val v = gc(up)
      so.times(g, ops.map(v)(u => rf.grad(gc, u)))
    }
  }

  override def toString: String = {
    rf.name + "(" + up + ")"
  }
}
