package pp

import scalaz.Scalaz.Id

package object ad {

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

    def apply[U[_], V, S](node: Node[U, V, S])(implicit vt: ValueOps[U, V, S], ops: ContainerOps.Aux[U, S], impl: Impl[V]): Node[U, V, S] =
      UnaryNode(node, impl)

    def wrapFunc[V](fn: UnaryValueFunc[V]): Impl[V] = new Impl[V] {

      override def apply(v: V) = fn.apply(v)

      override def grad(v: V) = fn.grad(v)
    }

    trait Impl[V] extends UnaryValueFunc[V]

  }

  trait CollectNodeFunc {

    def apply[U[_], V, S](node: Node[U, V, S])(implicit vt: ValueOps[U, V, S], idT: ValueOps[Id, V, Any], ops: ContainerOps[U], impl: Impl[V]): Node[Id, V, Any] =
      AccumulatingNode(node, impl)

    def wrapFunc[V](fn: CollectValueFunc[V]): Impl[V] = new Impl[V] {

      override def apply(acc: V, v: V) = fn.apply(acc, v)

      override def start = fn.start

      override def grad(acc: V, v: V) = fn.grad(acc, v)
    }

    trait Impl[V] extends CollectValueFunc[V]

  }

  trait Model {

    def valueOf[U[_], V, S](v: Variable[U, V, S])(implicit vo: ValueOps[U, V, S], ops: ContainerOps.Aux[U, S]): U[V]
  }

  object log extends UnaryNodeFunc {

    implicit def impl[V](implicit numV: Floating[V]): Impl[V] = wrapFunc(numV.log)
  }

  object sum extends CollectNodeFunc {

    implicit def impl[V](implicit numV: Floating[V]): Impl[V] = wrapFunc(numV.sum)
  }

  implicit def nodeNumeric[U[_], V, S](implicit vOps: ValueOps[U, V, S], cOps: ContainerOps.Aux[U, S]): Numeric[Node[U, V, S]] = new Numeric[Node[U, V, S]] {

    override def plus(x: Node[U, V, S], y: Node[U, V, S]) = x + y

    override def minus(x: Node[U, V, S], y: Node[U, V, S]) = x - y

    override def times(x: Node[U, V, S], y: Node[U, V, S]) = x * y

    override def negate(x: Node[U, V, S]) = -x

    override def fromInt(x: Int) = Constant(cOps.lift(vOps.valueVT.fromInt(x)))

    override def toInt(x: Node[U, V, S]) = ???

    override def toLong(x: Node[U, V, S]) = ???

    override def toFloat(x: Node[U, V, S]) = ???

    override def toDouble(x: Node[U, V, S]) = ???

    override def compare(x: Node[U, V, S], y: Node[U, V, S]) = ???
  }

}
