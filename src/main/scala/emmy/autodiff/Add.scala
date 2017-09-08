package emmy.autodiff

case class Add[U[_], V, S](lhs: Expression[U, V, S], rhs: Expression[U, V, S])
                          (implicit
                           val vt: ValueOps[U, V, S],
                           val ops: ContainerOps.Aux[U, S])
  extends Expression[U, V, S] {

  assert(lhs.shape == rhs.shape)

  override val shape = lhs.shape

  override val parents = Seq(lhs, rhs)

  override def apply(ec: EvaluationContext) =
      vt.plus(ec(lhs), ec(rhs))

  override def grad[W[_], T](gc: GradientContext, v: Variable[W, V, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val ops = implicitly[ContainerOps[W]]
    ops.zipMap(gc(lhs, v), gc(rhs, v)) {
      (lg, rg) => vt.plus(lg, rg)
    }
  }

}
