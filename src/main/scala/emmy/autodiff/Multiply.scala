package emmy.autodiff

case class Multiply[U[_], V, S](lhs: Expression[U, V, S], rhs: Expression[U, V, S])
                               (implicit
                                val vt: Evaluable[ValueOps[U, V, S]],
                                val so: ScalarOps[U[Double], U[V]],
                                val ops: ContainerOps.Aux[U, S])
  extends Expression[U, V, S] {

  override val parents = Seq(lhs, rhs)

  override def apply(ec: EvaluationContext) = {
    vt(ec).times(ec(lhs), ec(rhs))
  }

  override def grad[W[_], T](gc: GradientContext, v: Variable[W, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val ops = implicitly[ContainerOps[W]]
    val lv = gc(lhs)
    val leftg = gc(lhs, v)
    val rv = gc(rhs)
    val rightg = gc(rhs, v)
    val valT = vt(gc).forDouble
    ops.zipMap(leftg, rightg) {
      (lg, rg) =>
        valT.plus(
          so.times(lg, rv),
          so.times(rg, lv)
        )
    }
  }

  override def toString: String = {
    "(" + lhs + " * " + rhs + ")"
  }

}
