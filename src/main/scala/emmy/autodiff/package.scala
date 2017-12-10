package emmy

import scalaz.Scalaz.Id

package object autodiff {

  type Gradient[W[_], A[_]] = Option[Evaluable[W[A[Double]]]]

  implicit def toNode[U[_], V, S](value: U[V])(implicit
                                               fl: Floating[V],
                                               so:  ScalarOps[U[Double], U[V]],
                                               ops: ContainerOps.Aux[U, S]
  ): Expression[U, V, S] = {
    Constant[U, V, S](value)
  }

  implicit def toIdNode[V](value: V)(implicit
                                     fl: Floating[V],
                                     so:  ScalarOps[Double, V],
                                     ops: ContainerOps.Aux[Id, Any]
  ): Expression[Id, V, Any] = {
    Constant[Id, V, Any](value)
  }

  implicit def liftContainer[U[_], V, S](value: U[Expression[Id, V, Any]])(implicit
                                                                           fl: Floating[V],
                                                                           soo:  ScalarOps[Double, V],
                                                                           opso: ContainerOps.Aux[U, S]
  ): Expression[U, V, S] = {
    LiftedContainer[U, V, S](value)
  }

}
