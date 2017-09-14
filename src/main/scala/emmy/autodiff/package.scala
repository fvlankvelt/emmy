package emmy

import scalaz.Scalaz.Id

package object autodiff {

  type Gradient[W[_], A[_], V] = W[A[V]]

  trait UnaryValueFunc[V] extends (V => V) {

    def grad(v: V): V
  }

  trait CollectValueFunc[V] extends ((V, V) => V) {

    def start: V

    // gradient of the accumulator to v at a
    def grad(a: V, v: V): V
  }

  trait UnaryNodeFunc {

    def apply[U[_], V, S](node: Expression[U, V, S])
                         (implicit impl: Impl[V]): Expression[U, V, S] =
      UnaryExpression(node, impl)

    def wrapFunc[V](fn: UnaryValueFunc[V]): Impl[V] = new Impl[V] {

      override def apply(v: V) = fn.apply(v)

      override def grad(v: V) = fn.grad(v)
    }

    trait Impl[V] extends UnaryValueFunc[V]

  }

  trait CollectNodeFunc {

    def apply[U[_], V, S](node: Expression[U, V, S])
                         (implicit
                          vt: ValueOps[U, V, S],
                          idT: ValueOps[Id, V, Any],
                          ops: ContainerOps[U],
                          impl: Impl[V]): Expression[Id, V, Any] =
      AccumulatingExpression(node, impl)

    def wrapFunc[V](fn: CollectValueFunc[V]): Impl[V] = new Impl[V] {

      override def apply(acc: V, v: V) = fn.apply(acc, v)

      override def start = fn.start

      override def grad(acc: V, v: V) = fn.grad(acc, v)
    }

    trait Impl[V] extends CollectValueFunc[V]

  }

  trait EvaluationContext[V] {

    def apply[U[_], S](n: Expression[U, V, S]): U[V]
  }

  trait GradientContext[V] extends EvaluationContext[V] {

    def apply[W[_], U[_], T, S](n: Expression[U, V, S], v: Variable[W, V, T])(implicit wOps: ContainerOps.Aux[W, T]): W[U[V]]
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

  object sum extends CollectNodeFunc {

    implicit def impl[V](implicit numV: Floating[V]): Impl[V] = wrapFunc(numV.sum)
  }

  implicit def nodeNumeric[U[_], V, S](implicit
                                       vOps: ValueOps[U, V, S],
                                       cOps: ContainerOps.Aux[U, S]): Numeric[Expression[U, V, S]] = new Numeric[Expression[U, V, S]] {

    override def plus(x: Expression[U, V, S], y: Expression[U, V, S]) = x + y

    override def minus(x: Expression[U, V, S], y: Expression[U, V, S]) = x - y

    override def times(x: Expression[U, V, S], y: Expression[U, V, S]) = x * y

    override def negate(x: Expression[U, V, S]) = -x

    override def fromInt(x: Int) = Constant(cOps.lift(vOps.valueVT.fromInt(x)))

    override def toInt(x: Expression[U, V, S]) = ???

    override def toLong(x: Expression[U, V, S]) = ???

    override def toFloat(x: Expression[U, V, S]) = ???

    override def toDouble(x: Expression[U, V, S]) = ???

    override def compare(x: Expression[U, V, S], y: Expression[U, V, S]) = ???
  }

  implicit def toNode[U[_], V, S](value: U[V])
                                 (implicit
                                  vo: ValueOps[U, V, S],
                                  ops: ContainerOps.Aux[U, S]): Expression[U, V, S] = {
    Constant[U, V, S](value)
  }

  implicit def toIdNode[V](value: V)
                          (implicit
                           vo: ValueOps[Id, V, Any],
                           ops: ContainerOps.Aux[Id, Any]): Expression[Id, V, Any] = {
    Constant[Id, V, Any](value)
  }

  implicit class RichScalar[W, U[_], V, S](value: W) {

    def -(node: Expression[U, V, S])(implicit sOps: ScalarOps[V, W]): Expression[U, V, S] = {
      -node + value
    }

    def +(node: Expression[U, V, S])(implicit sOps: ScalarOps[V, W]): Expression[U, V, S] = {
      node + value
    }

    def *(node: Expression[U, V, S])(implicit sOps: ScalarOps[V, W]): Expression[U, V, S] = {
      node * value
    }

    def /(node: Expression[U, V, S])(implicit sOps: ScalarOps[V, W]): Expression[U, V, S] = {
      value * node.reciprocal()
    }
  }

}
