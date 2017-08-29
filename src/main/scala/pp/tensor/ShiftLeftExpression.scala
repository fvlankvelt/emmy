package pp.tensor

import breeze.linalg.View

case class ShiftLeftExpression[
K <: Nat,
CK <: Nat,
L <: Nat : ToInt
](self: Expression[K, CK]) extends Expression[Plus[K, L], Min[CK, L]] {

  override val shape = self.shape.shiftLeft[L]

  override def eval() = {
    val tensor = self.eval()
    val data = tensor.data
    val reshaped = data.reshape(shape.dom.size, shape.mod.size, View.Require)
    Tensor[Plus[K, L], Min[CK, L]](shape.dom, shape.mod, reshaped)
  }

  override def grad[M <: Nat : ToInt](variable: Variable[M]) = {
    val upstream: Expression[Plus[Min[Plus[K, M], M], L], Plus[Min[Min[Plus[M, CK], M], L], M]] =
      self.grad(variable)
        .shiftLeft[M]
        .transpose[M, L]
    upstream.asInstanceOf[Expression[Plus[K, L], Plus[M, Min[CK, L]]]]
  }

}

