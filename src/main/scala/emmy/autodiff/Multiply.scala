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

  override def eval(ec: GradientContext) = {
    val cLhs = ec(lhs)
    val cRhs = ec(rhs)
    ctx ⇒ {
      vt(ctx).times(cLhs(ctx), cRhs(ctx))
    }
  }

  override def grad[W[_], T](gc: GradientContext, v: Parameter[W, T]) = {
    val wOps = v.ops
    val rv = gc(rhs)
    val lv = gc(lhs)
    (gc(lhs, v), gc(rhs, v)) match {
      case (None, None) ⇒ None
      case (Some(leftg), None) ⇒
        Some { ctx ⇒
          wOps.map(leftg(ctx)) { lg ⇒
            so.times(lg, rv(ctx))
          }
        }
      case (None, Some(rightg)) ⇒
        Some { ctx ⇒
          wOps.map(rightg(ctx)) { rg ⇒
            so.times(rg, lv(ctx))
          }
        }
      case (Some(leftg), Some(rightg)) ⇒
        Some { ctx ⇒
          val valT = vt(ctx).forDouble
          wOps.zipMap(leftg(ctx), rightg(ctx)) {
            (lg, rg) ⇒
              valT.plus(
                so.times(lg, rv(ctx)),
                so.times(rg, lv(ctx))
              )
          }
        }
    }
  }

  override def toString: String = {
    "(" + lhs + " * " + rhs + ")"
  }

}
