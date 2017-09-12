package emmy.autodiff

case class Reciprocal[U[_], V, S](upstream: Expression[U, V, S])
                                 (implicit
                                  val vt: ValueOps[U, V, S],
                                  val ops: ContainerOps.Aux[U, S])
  extends Expression[U, V, S] {

  override val shape = upstream.shape

  override val parents = Seq(upstream)

  override def apply(ec: EvaluationContext[V]) = {
    vt.div(vt.one, ec(upstream))
  }

  override def grad[W[_], T](gc: GradientContext[V], v: Variable[W, V, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val ops = implicitly[ContainerOps[W]]
    val value = gc(upstream)
    val grad = gc(upstream, v)
    ops.map(grad) { g =>
      vt.div(g, vt.times(value, value))
    }
  }

}
