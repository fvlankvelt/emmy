package pp.tensor

import breeze.math.Field

import scala.reflect.ClassTag

case class DivExpression[
K <: Nat,
CK <: Nat
](left: Expression[K, CK], right: Expression[K, CK])
  extends Expression[K, CK] {

  assert(left.shape == right.shape)

  override val shape = left.shape

  override def eval() = {
    val leftTensor = left.eval()
    val rightTensor = right.eval()
    Tensor[K, CK](shape.dom, shape.mod, leftTensor.data /:/ rightTensor.data)
  }

  override def grad[M <: Nat : ToInt](variable: Variable[M]) = {
    val leftGrad: Expression[K, Plus[M, CK]] = left.grad(variable)
    val rightGrad: Expression[K, Plus[M, CK]] = right.grad(variable)

    leftGrad / right.broadcastCov[M](variable.dom) -
      left.broadcastCov[M](variable.dom) * rightGrad / (right * right).broadcastCov[M](variable.dom)
  }
}

