package emmy.autodiff

case class Add[U[_], V, S](
    lhs: Expression[U, V, S],
    rhs: Expression[U, V, S]
)(implicit
    val vt: Evaluable[ValueOps[U, V, S]],
  val so:  ScalarOps[U[Double], U[V]],
  val ops: ContainerOps.Aux[U, S]
)
  extends Expression[U, V, S] {

  override val parents = Seq(lhs, rhs)

  override def apply(ec: EvaluationContext) = {
    val valT = vt(ec)
    valT.plus(ec(lhs), ec(rhs))
  }

  override def grad[W[_], T](gc: GradientContext, v: ContinuousVariable[W, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val valT = vt(gc)
    val ring = valT.forDouble
    wOps.zipMap(gc(lhs, v), gc(rhs, v)) {
      (lg, rg) ⇒ ring.plus(lg, rg)
    }
  }

  override def toString: String = {
    "(" + lhs + " + " + rhs + ")"
  }

}
