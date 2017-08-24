package pp.tensor

import breeze.linalg.View
import breeze.math.Semiring

import scala.reflect.ClassTag

case class ShiftLeftExpression[
V: ClassTag : Semiring,
K <: Nat,
CK <: Nat,
L <: Nat : ToInt
](self: Expression[V, K, CK]) extends Expression[V, Plus[K, L], Min[CK, L]] {

  val ringV = self.ringV
  val ctV = self.ctV

  override val shape = self.shape.shiftLeft[L]

  override def eval() = {
    val tensor = self.eval()
    val data = tensor.data
    val reshaped = data.reshape(shape.dom.size, shape.mod.size, View.Require)
    Tensor[V, Plus[K, L], Min[CK, L]](shape.dom, shape.mod, reshaped)
  }

  override def grad[M <: Nat : ToInt](variable: Variable[V, M]) = {
    val upstream: Expression[V, Plus[Min[Plus[K, M], M], L], Plus[Min[Min[Plus[M, CK], M], L], M]] =
      self.grad(variable)
        .shiftLeft[M]
        .transpose[M, L]
    upstream.asInstanceOf[Expression[V, Plus[K, L], Plus[M, Min[CK, L]]]]
  }

}

