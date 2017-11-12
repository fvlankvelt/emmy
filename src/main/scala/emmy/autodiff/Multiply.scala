package emmy.autodiff

case class Multiply[U[_], V, S](
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
    vt(ec).times(ec(lhs), ec(rhs))
  }

  override def grad[W[_], T](gc: GradientContext, v: ContinuousVariable[W, T])(implicit wOps: ContainerOps.Aux[W, T]) =
    (gc(lhs, v), gc(rhs, v)) match {
      case (None, None) ⇒ None
      case (Some(leftg), None) ⇒
        val rv = gc(rhs)
        Some(wOps.map(leftg) { lg ⇒
          so.times(lg, rv)
        })
      case (None, Some(rightg)) ⇒
        val lv = gc(lhs)
        Some(wOps.map(rightg) { rg ⇒
          so.times(rg, lv)
        })
      case (Some(leftg), Some(rightg)) ⇒
        val lv = gc(lhs)
        val rv = gc(rhs)
        val valT = vt(gc).forDouble
        Some(wOps.zipMap(leftg, rightg) {
          (lg, rg) ⇒
            valT.plus(
              so.times(lg, rv),
              so.times(rg, lv)
            )
        })
    }

  override def toString: String = {
    "(" + lhs + " * " + rhs + ")"
  }

}
