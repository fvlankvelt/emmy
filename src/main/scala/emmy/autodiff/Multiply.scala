package emmy.autodiff

case class Multiply[U[_], V, S](lhs: Expression[U, V, S], rhs: Expression[U, V, S])
                               (implicit
                                val vt: Evaluable[ValueOps[U, V, S]],
                                val ops: ContainerOps.Aux[U, S])
  extends Expression[U, V, S] {

  override val parents = Seq(lhs, rhs)

  override def apply(ec: EvaluationContext[V]) = {
    vt(ec).times(ec(lhs), ec(rhs))
  }

  override def grad[W[_], T](gc: GradientContext[V], v: Variable[W, V, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val ops = implicitly[ContainerOps[W]]
    val lv = gc(lhs)
    val leftg = gc(lhs, v)
    val rv = gc(rhs)
    val rightg = gc(rhs, v)
    val valT = vt(gc)
    ops.zipMap(leftg, rightg) {
      (lg, rg) =>
        valT.plus(
          valT.times(lg, rv),
          valT.times(lv, rg)
        )
    }
  }

  override def toString: String = {
    "(" + lhs + " * " + rhs + ")"
  }

}
