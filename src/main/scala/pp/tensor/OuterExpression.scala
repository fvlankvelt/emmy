package pp.tensor

import breeze.linalg.DenseMatrix

case class OuterExpression[
KL <: Nat,
CKL <: Nat,
KR <: Nat,
CKR <: Nat
](left: Expression[KL, CKL], right: Expression[KR, CKR])
  extends Expression[Plus[KL, KR], Plus[CKR, CKL]] {

  override val shape = left.shape.outer(right.shape)

  override def eval() = {
    val leftMatrix = left.eval().data
    val rightMatrix = right.eval().data
    val newMatrix = DenseMatrix.zeros[Float](shape.dom.size, shape.mod.size)
    for {
      rowLeft <- 0 until left.shape.dom.size
      rowRight <- 0 until right.shape.dom.size
      colLeft <- 0 until left.shape.mod.size
      colRight <- 0 until right.shape.mod.size
    } {
      newMatrix(
        rowLeft * right.shape.dom.size + rowRight,
        colLeft * right.shape.mod.size + colRight
      ) = leftMatrix(rowLeft, colLeft) * rightMatrix(rowRight, colRight)
    }
    Tensor(shape.dom, shape.mod, newMatrix)
  }

  override def grad[M <: Nat : ToInt](variable: Variable[M]) = {
    val leftGrad = left.grad(variable)
      .asInstanceOf[Expression[KL, Plus[M, CKL]]]
    val rightGrad = right.grad(variable)

    implicit val ckToInt = right.shape.dom.toInt
    implicit val ckrToInt = right.shape.mod.toInt
    val lor = (leftGrad outer right)
      .shiftLeft[CKR]
      .transpose[CKR, M]
      .shiftRight[M]
    val leftRes: Expression[Plus[KL, KR], Plus[M, Plus[CKR, CKL]]] =
      lor.asInstanceOf[Expression[Plus[KL, KR], Plus[M, Plus[CKR, CKL]]]]

    implicit val mpckr = new ToInt[Plus[M, CKR]] {
      override def apply() = implicitly[ToInt[M]].apply() + ckrToInt.apply()
    }
    val rightRes = (left outer rightGrad)
      .asInstanceOf[Expression[Plus[KL, KR], Plus[M, Plus[CKR, CKL]]]]

    leftRes + rightRes
  }

}

