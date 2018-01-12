package emmy.autodiff

trait UnaryValueFunc[V] extends (V ⇒ V) {

  def name: String

  def grad(v: V): V
}

trait EvaluableValueFunc[V] {

  def name: String

  def apply(ec: SampleContext, v: V): V

  def grad(gc: SampleContext, v: V): V
}

object EvaluableValueFunc {
  implicit def fromUnary[V](rf: UnaryValueFunc[V]) = new EvaluableValueFunc[V] {

    override def name = rf.name

    override def apply(ec: SampleContext, v: V) = rf(v)

    override def grad(gc: SampleContext, v: V) = rf.grad(v)
  }
}

trait UnaryNodeFunc {

  def apply[U[_], V, S](node: Expression[U, V, S])(implicit impl: Impl[V]): Expression[U, V, S] =
    UnaryExpression(node, impl)

  def wrapFunc[V](fn: EvaluableValueFunc[V]): Impl[V] = new Impl[V] {

    override def name: String = fn.name

    override def apply(ec: SampleContext, v: V) = fn.apply(ec, v)

    override def grad(gc: SampleContext, v: V) = fn.grad(gc, v)
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

object softmax extends UnaryNodeFunc {

  implicit def impl[V](implicit numV: Floating[V]): Impl[V] = wrapFunc(numV.softmax)
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

  override def eval(ec: GradientContext) = {
    val value = ec(up)
    ctx ⇒ {
      ops.map(value(ctx))(v ⇒ rf.apply(ctx, v))
    }
  }

  override def grad[W[_], T](gc: GradientContext, v: Parameter[W, T]) = {
    val wOps = v.ops
    val value = gc(up)
    gc(up, v).map { upGrad ⇒ ctx ⇒ {
      val ug = upGrad(ctx)
      val uv = value(ctx)
      wOps.map(ug) { g ⇒
        so.times(g, ops.map(uv)(u ⇒ rf.grad(ctx, u)))
      }
    }
    }
  }

  override def toString: String = {
    rf.name + "(" + up + ")"
  }
}
