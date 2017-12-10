package emmy.autodiff

case class Reciprocal[U[_], V, S](
    upstream: Expression[U, V, S]
)(implicit
    val vt: Evaluable[ValueOps[U, V, S]],
  val so:  ScalarOps[U[Double], U[V]],
  val ops: ContainerOps.Aux[U, S]
)
  extends Expression[U, V, S] {

  override val parents = Seq(upstream)

  override def eval(ec: GradientContext) = {
    val upstreamValue = ec(upstream)
    ctx => {
      val lvt = vt(ctx)
      lvt.div(lvt.one, upstreamValue(ctx))
    }
  }

  override def grad[W[_], T](gc: GradientContext, v: Parameter[W, T]) = {
    val wOps = v.ops
    val value = gc(upstream)
    gc(upstream, v).map { grad ⇒
      ctx => {
        val valT = vt(ctx)
        val valD = valT.forDouble
        val myval = value(ctx)
        wOps.map(grad(ctx)) { g ⇒
          valD.negate(so.div(g, valT.times(myval, myval)))
        }
      }
    }
  }

  override def toString: String = {
    "(1 / " + upstream + ")"
  }

}
