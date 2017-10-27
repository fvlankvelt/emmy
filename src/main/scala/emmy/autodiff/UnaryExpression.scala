package emmy.autodiff

trait UnaryValueFunc[V] extends (V ⇒ V) {

  def name: String

  def grad(v: V): V
}

trait EvaluableValueFunc[V] {

  def name: String

  def apply(ec: EvaluationContext, v: V): V

  def grad(gc: GradientContext, v: V): V
}

object EvaluableValueFunc {
  implicit def fromUnary[V](rf: UnaryValueFunc[V]) = new EvaluableValueFunc[V] {

    override def name = rf.name

    override def apply(ec: EvaluationContext, v: V) = rf(v)

    override def grad(gc: GradientContext, v: V) = rf.grad(v)
  }
}

trait UnaryNodeFunc {

  def apply[U[_], V, S](node: Expression[U, V, S])(implicit impl: Impl[V]): Expression[U, V, S] =
    UnaryExpression(node, impl)

  def wrapFunc[V](fn: EvaluableValueFunc[V]): Impl[V] = new Impl[V] {

    override def name: String = fn.name

    override def apply(ec: EvaluationContext, v: V) = fn.apply(ec, v)

    override def grad(gc: GradientContext, v: V) = fn.grad(gc, v)
  }

  trait Impl[V] extends EvaluableValueFunc[V]

}

object log extends UnaryNodeFunc {

  implicit def impl[V](implicit numV: Floating[V]): Impl[V] = wrapFunc(numV.log)
}

object exp extends UnaryNodeFunc {

  implicit def impl[V](implicit numV: Floating[V]): Impl[V] = wrapFunc(numV.exp)
}

object lgamma extends UnaryNodeFunc {

  implicit def impl[V](implicit numV: Floating[V]): Impl[V] = wrapFunc(numV.lgamma)
}

case class UnaryExpression[U[_], V, S](
    up: Expression[U, V, S],
    rf: EvaluableValueFunc[V]
)
  extends Expression[U, V, S] {

  override val vt = up.vt

  override val ops = up.ops

  override val so = up.so

  override val parents = Seq(up)

  override def apply(ec: EvaluationContext) = {
    val value = ec(up)
    ops.map(value)(v ⇒ rf.apply(ec, v))
  }

  override def grad[W[_], T](gc: GradientContext, v: ContinuousVariable[W, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val opsW = implicitly[ContainerOps[W]]
    val ug = gc(up, v)
    opsW.map(ug) { g ⇒
      val v = gc(up)
      so.times(g, ops.map(v)(u ⇒ rf.grad(gc, u)))
    }
  }

  override def toString: String = {
    rf.name + "(" + up + ")"
  }
}
