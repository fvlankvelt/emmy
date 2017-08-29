package pp.tensor

import breeze.linalg.View

case class TransposeExpression[
K <: Nat,
CK <: Nat,
L <: Nat : ToInt,
CL <: Nat : ToInt
](orig: Expression[K, CK])
  extends Expression[Plus[Min[K, L], CL], Plus[Min[CK, CL], L]] {

  override val shape = orig.shape.transpose[L, CL]

  override def eval() = {
    val tensor = orig.eval()
    val data = tensor.data

    val l = implicitly[ToInt[L]].apply()
    val cl = implicitly[ToInt[CL]].apply()
    val blockRows = orig.shape.dom.sizes.takeRight(l).product
    val blockCols = orig.shape.mod.sizes.take(cl).product
    val newData = data.reshape(shape.dom.size, shape.mod.size, View.Copy)
    for {
      row <- 0 until orig.shape.dom.sizes.dropRight(l).product
      col <- 0 until orig.shape.mod.sizes.drop(cl).product
    } {
      val view = data(
        (row * blockRows) until ((row + 1) * blockRows),
        (col * blockCols) until ((col + 1) * blockCols)
      )
      newData(
        (row * blockCols) until ((row + 1) * blockCols),
        (col * blockRows) until ((col + 1) * blockRows)
      ) := view.t
    }
    Tensor[Plus[Min[K, L], CL], Plus[Min[CK, CL], L]](shape.dom, shape.mod, newData)
  }

  override def grad[M <: Nat : ToInt](variable: Variable[M]) = {
    val upstream = orig.grad(variable)
    val result = upstream.transpose[L, M]
      .shiftLeft[L]
      .transpose[L, CL]
      .shiftRight[CL]
      .transpose[M, CL]
    result.asInstanceOf[Expression[Plus[Min[K, L], CL], Plus[M, Plus[Min[CK, CL], L]]]]
  }
}
