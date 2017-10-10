package emmy.autodiff

case class Add[U[_], V, S](lhs: Expression[U, V, S], rhs: Expression[U, V, S])
                          (implicit
                           val vt: Evaluable[ValueOps[U, V, S]],
                           val ops: ContainerOps.Aux[U, S])
  extends Expression[U, V, S] {

  override val parents = Seq(lhs, rhs)

  override def apply(ec: EvaluationContext[V]) = {
    val valT = vt(ec)
    valT.plus(ec(lhs), ec(rhs))
  }

  override def grad[W[_], T](gc: GradientContext[V], v: Variable[W, V, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val ops = implicitly[ContainerOps[W]]
    val valT = vt(gc)
    ops.zipMap(gc(lhs, v), gc(rhs, v)) {
      (lg, rg) => valT.plus(lg, rg)
    }
  }

  override def toString: String = {
    "(" + lhs + " + " + rhs + ")"
  }

}
