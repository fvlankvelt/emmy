package emmy.autodiff

import scalaz.Scalaz.Id

trait CollectValueFunc[V] extends ((V, V) ⇒ V) {

  def name: String

  def start: V

  // gradient of the accumulator to v at a
  def grad(a: V, v: V): V
}

trait CollectNodeFunc {

  def apply[U[_], V, S](node: Expression[U, V, S])(implicit
    fl: Floating[V],
                                                   so:   ScalarOps[Double, V],
                                                   ops:  ContainerOps[U],
                                                   impl: Impl[V]
  ): Expression[Id, V, Any] =
    AccumulatingExpression(node, impl)

  def wrapFunc[V](fn: CollectValueFunc[V]): Impl[V] = new Impl[V] {

    override def name: String = fn.name

    override def apply(acc: V, v: V) = fn.apply(acc, v)

    override def start = fn.start

    override def grad(acc: V, v: V) = fn.grad(acc, v)
  }

  trait Impl[V] extends CollectValueFunc[V]

}

object sum extends CollectNodeFunc {

  implicit def impl[V](implicit numV: Floating[V]): Impl[V] = wrapFunc(numV.sum)
}

case class AccumulatingExpression[U[_]: ContainerOps, V, S, A](
    up: Expression[U, V, S],
    rf: CollectValueFunc[V]
)(implicit
    fl: Floating[V],
  val so: ScalarOps[Double, V]
)
  extends Expression[Id, V, Any] {

  override val ops = ContainerOps.idOps

  override val vt = Evaluable.fromConstant(ValueOps[Id, V, Any](fl, ContainerOps.idOps, null))

  private val opsU = implicitly[ContainerOps[U]]

  override val parents = Seq(up)

  override def eval(ec: GradientContext) = {
    ec(up).map { e =>
      opsU.foldLeft(e)(rf.start)(rf.apply)
    }
  }

  // f(f(f(zero, x1), x2), x3)
  // grad_v =>
  //   (
  //    x3' +
  //    (
  //     x2' +
  //     (x1' * f'(zero, x1))
  //    ) * f'(f(zero, x1), x2) +
  //   ) * f'(f(f(zero, x1), x2), x3)

  // ug = (x1', x2', x3')

  override def grad[W[_], T](gc: GradientContext, v: Parameter[W, T]) = {
    val wOps = v.ops
    val upValue = gc(up)
    gc(up, v).map { upGrad ⇒
      ctx => {
        val valT = vt(ctx)
        val ug = upGrad(ctx)
        implicit val sod = so
        val valD = valT.forDouble
        val result = wOps.map(ug) { g ⇒
          val vg = opsU.zipMap(upValue(ctx), g)((_, _))
          opsU.foldLeft(vg)((rf.start, valD.zero)) {
            (acc, x) ⇒
              val (av, ag) = acc
              val (xv, xg) = x
              (
                rf(av, xv),
                sod.times(valD.plus(xg, ag), rf.grad(av, xv))
              )
          }
        }
        wOps.map(result)(_._2)
      }
    }
  }

  override def toString: String = {
    rf.name + "(" + up + ")"
  }
}
