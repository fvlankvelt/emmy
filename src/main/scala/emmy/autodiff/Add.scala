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

  override def eval(ec: GradientContext) = {
    val l = ec(lhs)
    val r = ec(rhs)
    ctx => {
      val valT = vt(ctx)
      valT.plus(l(ctx), r(ctx))
    }
  }

  override def grad[W[_], T](gc: GradientContext, v: Parameter[W, T]) = {
    val wOps = v.ops
    (gc(lhs, v), gc(rhs, v)) match {
      case (None, None) ⇒ None
      case (Some(lg), None) ⇒ Some(lg)
      case (None, Some(rg)) ⇒ Some(rg)
      case (Some(lg), Some(rg)) ⇒
        Some { ctx =>
          val valT = vt(ctx)
          val ring = valT.forDouble
          wOps.zipMap(lg(ctx), rg(ctx)) {
            (l, r) ⇒ ring.plus(l, r)
          }
        }
    }
  }

  override def toString: String = {
    "(" + lhs + " + " + rhs + ")"
  }

}
