package pp.tensor

import breeze.linalg.DenseMatrix
case class Tensor[DOM <: Nat, MOD <: Nat](dom: Domain[DOM], mod: Domain[MOD], data: DenseMatrix[Float]) {

  def transpose: Tensor[MOD, DOM] = Tensor(mod, dom, data.t)
}

object Tensor {

  def apply[DOM <: Nat, MOD <: Nat](dom: Domain[DOM], mod: Domain[MOD]): Tensor[DOM, MOD] =
    Tensor[DOM, MOD](dom, mod, DenseMatrix.zeros[Float](dom.size, mod.size))

  def ones[DOM <: Nat, MOD <: Nat](dom: Domain[DOM], mod: Domain[MOD]) =
    Tensor[DOM, MOD](dom, mod, DenseMatrix.ones[Float](dom.size, mod.size))

  def eye[DOM <: Nat](dom: Domain[DOM]) = {
    Tensor[DOM, DOM](dom, dom, DenseMatrix.eye[Float](dom.size))
  }

}

