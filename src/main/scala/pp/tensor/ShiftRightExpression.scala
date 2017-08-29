package pp.tensor

import breeze.linalg.View

case class ShiftRightExpression[
K <: Nat,
CK <: Nat,
L <: Nat : ToInt
](self: Expression[K, CK])
  extends Expression[Min[K, L], Plus[CK, L]] {

  override val shape = self.shape.shiftRight[L]

  override def eval() = {
    val tensor = self.eval()
    val data = tensor.data
    val reshaped = data.reshape(shape.dom.size, shape.mod.size, View.Require)
    Tensor[Min[K, L], Plus[CK, L]](shape.dom, shape.mod, reshaped)
  }

  override def grad[M <: Nat : ToInt](variable: Variable[M]) = {
    val upstream = self.grad(variable)
      .transpose[L, M]
      .shiftRight[M]
    upstream.asInstanceOf[Expression[Min[K, L], Plus[M, Plus[CK, L]]]]
  }
}

