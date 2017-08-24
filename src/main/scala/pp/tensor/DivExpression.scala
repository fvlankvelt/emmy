package pp.tensor

import breeze.math.Semiring

import scala.reflect.ClassTag

case class DivExpression[
V: ClassTag : Semiring,
K <: Nat,
CK <: Nat
](left: Expression[V, K, CK], right: Expression[V, K, CK])
  extends Expression[V, K, CK] {

  assert(left.shape == right.shape)

  val ringV = implicitly[Semiring[V]]
  val ctV = implicitly[ClassTag[V]]

  override val shape = left.shape

  override def eval() = {
    val leftTensor = left.eval()
    val rightTensor = right.eval()
    new Tensor[V, K, CK](shape.dom, shape.mod, leftTensor.data /:/ rightTensor.data)
  }

  override def grad[M <: Nat : ToInt](variable: Variable[V, M]) = {
    val leftGrad: Expression[V, K, Plus[M, CK]] = left.grad(variable)
    val rightGrad: Expression[V, K, Plus[M, CK]] = right.grad(variable)

    leftGrad / right.broadcast[Nat._0, M](Domain(), variable.dom).asInstanceOf[Expression[V, K, Plus[M, CK]]] -
      left.broadcast[Nat._0, M](Domain(), variable.dom).asInstanceOf[Expression[V, K, Plus[M, CK]]] * rightGrad /
        (right * right).broadcast[Nat._0, M](Domain(), variable.dom).asInstanceOf[Expression[V, K, Plus[M, CK]]]
  }
}

