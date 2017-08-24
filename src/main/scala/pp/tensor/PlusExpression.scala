package pp.tensor

import breeze.math.Field

import scala.reflect.ClassTag

case class PlusExpression[
V: ClassTag : Field,
K <: Nat,
CK <: Nat
](left: Expression[V, K, CK], right: Expression[V, K, CK])
  extends Expression[V, K, CK] {
  assert(left.shape == right.shape)

  val ringV = implicitly[Field[V]]
  val ctV = implicitly[ClassTag[V]]

  override val shape = left.shape

  override def eval() = {
    val leftTensor = left.eval()
    val rightTensor = right.eval()
    Tensor[V, K, CK](shape.dom, shape.mod, leftTensor.data +:+ rightTensor.data)
  }

  override def grad[M <: Nat : ToInt](variable: Variable[V, M]) = {
    val leftGrad = left.grad(variable)
    val rightGrad = right.grad(variable)
    leftGrad + rightGrad
  }
}

