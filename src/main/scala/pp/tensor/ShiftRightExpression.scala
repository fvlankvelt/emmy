package pp.tensor

import breeze.linalg.View
import breeze.math.Semiring

import scala.reflect.ClassTag

case class ShiftRightExpression[
V: ClassTag : Semiring,
K <: Nat,
CK <: Nat,
L <: Nat : ToInt
](self: Expression[V, K, CK])
  extends Expression[V, Min[K, L], Plus[CK, L]] {

  val ringV = self.ringV
  val ctV = self.ctV

  override val shape = self.shape.shiftRight[L]

  override def eval() = {
    val tensor = self.eval()
    val data = tensor.data
    val reshaped = data.reshape(shape.dom.size, shape.mod.size, View.Require)
    Tensor[V, Min[K, L], Plus[CK, L]](shape.dom, shape.mod, reshaped)
  }

  override def grad[M <: Nat : ToInt](variable: Variable[V, M]) = {
    val upstream = self.grad(variable)
      .transpose[L, M]
      .shiftRight[M]
    upstream.asInstanceOf[Expression[V, Min[K, L], Plus[M, Plus[CK, L]]]]
  }
}

