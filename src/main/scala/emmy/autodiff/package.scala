package emmy

import scalaz.Scalaz.Id

package object autodiff {

  type Gradient[W[_], A[_]] = W[A[Double]]

  trait UnaryValueFunc[V] extends (V => V) {

    def name: String

    def grad(v: V): V
  }

  trait CollectValueFunc[V] extends ((V, V) => V) {

    def name: String

    def start: V

    // gradient of the accumulator to v at a
    def grad(a: V, v: V): V
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

    def apply[U[_], V, S](node: Expression[U, V, S])
                         (implicit impl: Impl[V]): Expression[U, V, S] =
      UnaryExpression(node, impl)

    def wrapFunc[V](fn: EvaluableValueFunc[V]): Impl[V] = new Impl[V] {

      override def name: String = fn.name

      override def apply(ec: EvaluationContext, v: V) = fn.apply(ec, v)

      override def grad(gc: GradientContext, v: V) = fn.grad(gc, v)
    }

    trait Impl[V] extends EvaluableValueFunc[V]

  }

  trait CollectNodeFunc {

    def apply[U[_], V, S](node: Expression[U, V, S])
                         (implicit
                          fl: Floating[V],
                          so: ScalarOps[Double, V],
                          ops: ContainerOps[U],
                          impl: Impl[V]): Expression[Id, V, Any] =
      AccumulatingExpression(node, impl)

    def wrapFunc[V](fn: CollectValueFunc[V]): Impl[V] = new Impl[V] {

      override def name: String = fn.name

      override def apply(acc: V, v: V) = fn.apply(acc, v)

      override def start = fn.start

      override def grad(acc: V, v: V) = fn.grad(acc, v)
    }

    trait Impl[V] extends CollectValueFunc[V]

  }

  trait EvaluationContext {

    def apply[U[_], V, S](n: Expression[U, V, S]): U[V]
  }

  trait GradientContext extends EvaluationContext {

    def apply[W[_], U[_], V, T, S](n: Expression[U, V, S], v: ContinuousVariable[W, T])(implicit wOps: ContainerOps.Aux[W, T]): W[U[Double]]
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

  implicit def toNode[U[_], V, S](value: U[V])
                                 (implicit
                                  fl: Floating[V],
                                  so: ScalarOps[U[Double], U[V]],
                                  ops: ContainerOps.Aux[U, S]): Expression[U, V, S] = {
    Constant[U, V, S](value)
  }

  implicit def toIdNode[V](value: V)
                          (implicit
                           fl: Floating[V],
                           so: ScalarOps[Double, V],
                           ops: ContainerOps.Aux[Id, Any]): Expression[Id, V, Any] = {
    Constant[Id, V, Any](value)
  }

  implicit def liftContainer[U[_], V, S](value: U[Expression[Id, V, Any]])
                                        (implicit
                                         fl: Floating[V],
                                         soo: ScalarOps[Double, V],
                                         opso: ContainerOps.Aux[U, S]): Expression[U, V, S] = {
    LiftedContainer[U, V, S](value)
  }

}
