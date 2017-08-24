package pp.tensor

import breeze.linalg.{*, BroadcastedColumns, BroadcastedRows, DenseMatrix, DenseVector}
import breeze.math.Semiring

import scala.reflect.ClassTag

class Tensor[V: Semiring : ClassTag, DOM <: Nat, MOD <: Nat](val dom: Domain[DOM], val mod: Domain[MOD], val data: DenseMatrix[V]) {

  def transpose: Tensor[V, MOD, DOM] = new Tensor(mod, dom, data.t)
}

object Tensor {

  def apply[V: Semiring : ClassTag, DOM <: Nat, MOD <: Nat](dom: Domain[DOM], mod: Domain[MOD]) =
    new Tensor[V, DOM, MOD](dom, mod, DenseMatrix.zeros[V](dom.size, mod.size))

  def ones[V: Semiring : ClassTag, DOM <: Nat, MOD <: Nat](dom: Domain[DOM], mod: Domain[MOD]) =
    new Tensor[V, DOM, MOD](dom, mod, DenseMatrix.ones[V](dom.size, mod.size))

  def eye[V: Semiring : ClassTag, DOM <: Nat](dom: Domain[DOM]) = {
    new Tensor[V, DOM, DOM](dom, dom, DenseMatrix.eye[V](dom.size))
  }

}

