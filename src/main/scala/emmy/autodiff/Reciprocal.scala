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

  override def apply(ec: EvaluationContext) = {
    vt(ec).div(vt(ec).one, ec(upstream))
  }

  override def grad[W[_], T](gc: GradientContext, v: ContinuousVariable[W, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    gc(upstream, v).map { grad ⇒
      val value = gc(upstream)
      val valT = vt(gc)
      val valD = valT.forDouble
      wOps.map(grad) { g ⇒
        valD.negate(so.div(g, valT.times(value, value)))
      }
    }
  }

  override def toString: String = {
    "(1 / " + upstream + ")"
  }

}
