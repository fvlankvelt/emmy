package pp.tensor

import breeze.linalg.DenseMatrix
import breeze.math.Semiring

import scala.reflect.ClassTag

case class Tensor[V: Semiring : ClassTag, DOM <: Nat, MOD <: Nat](dom: Domain[DOM], mod: Domain[MOD], data: DenseMatrix[V]) {

  def transpose: Tensor[V, MOD, DOM] = Tensor(mod, dom, data.t)
}

object Tensor {

  def apply[V: Semiring : ClassTag, DOM <: Nat, MOD <: Nat](dom: Domain[DOM], mod: Domain[MOD]) =
    Tensor[V, DOM, MOD](dom, mod, DenseMatrix.zeros[V](dom.size, mod.size))

  def ones[V: Semiring : ClassTag, DOM <: Nat, MOD <: Nat](dom: Domain[DOM], mod: Domain[MOD]) =
    Tensor[V, DOM, MOD](dom, mod, DenseMatrix.ones[V](dom.size, mod.size))

  def eye[V: Semiring : ClassTag, DOM <: Nat](dom: Domain[DOM]) = {
    Tensor[V, DOM, DOM](dom, dom, DenseMatrix.eye[V](dom.size))
  }

}

