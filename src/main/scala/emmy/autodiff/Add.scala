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
    (gc(lhs, v), gc(rhs, v)) match {
      case (None, None)     ⇒ None
      case (Some(lg), None) ⇒ Some(lg)
      case (None, Some(rg)) ⇒ Some(rg)
      case (Some(lg), Some(rg)) ⇒
        val valT = vt(gc)
        val ring = valT.forDouble
        Some(wOps.zipMap(lg, rg) {
          (l, r) ⇒ ring.plus(l, r)
        })
    }
  }

  override def toString: String = {
    "(" + lhs + " + " + rhs + ")"
  }

}
