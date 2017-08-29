package pp.tensor

case class MinExpression[
K <: Nat,
CK <: Nat
](left: Expression[K, CK], right: Expression[K, CK])
  extends Expression[K, CK] {
  assert(left.shape == right.shape)

  override val shape = left.shape

  override def eval() = {
    val leftTensor = left.eval()
    val rightTensor = right.eval()
    Tensor[K, CK](shape.dom, shape.mod, leftTensor.data -:- rightTensor.data)
  }

  override def grad[M <: Nat : ToInt](variable: Variable[M]) = {
    val leftGrad = left.grad(variable)
    val rightGrad = right.grad(variable)
    leftGrad - rightGrad
  }
}

