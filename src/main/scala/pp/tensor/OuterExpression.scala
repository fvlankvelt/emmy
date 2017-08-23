package pp.tensor

import breeze.linalg.DenseMatrix
import breeze.math.Semiring

import scala.reflect.ClassTag

case class OuterExpression[
V: ClassTag : Semiring,
KL <: Nat,
CKL <: Nat,
KR <: Nat,
CKR <: Nat
](left: Expression[V, KL, CKL], right: Expression[V, KR, CKR])
  extends Expression[V, Plus[KL, KR], Plus[CKR, CKL]] {

  val ringV = implicitly[Semiring[V]]
  val ctV = implicitly[ClassTag[V]]

  override val shape = left.shape.outer(right.shape)

  override def eval() = {
    val leftMatrix = left.eval().data
    val rightMatrix = right.eval().data
    val newMatrix = DenseMatrix.zeros[V](shape.dom.size, shape.mod.size)
    for {
      rowLeft <- 0 until left.shape.dom.size
      rowRight <- 0 until right.shape.dom.size
      colLeft <- 0 until left.shape.mod.size
      colRight <- 0 until right.shape.mod.size
    } {
      val entry = ringV.*(
        leftMatrix(rowLeft, colLeft),
        rightMatrix(rowRight, colRight)
      )
      newMatrix(
        rowLeft * right.shape.dom.size + rowRight,
        colLeft * right.shape.mod.size + colRight
      ) = entry
    }
    new Tensor(shape.dom, shape.mod, newMatrix)
  }

  override def grad[M <: Nat : ToInt](variable: Variable[V, M]) = {
    val leftGrad = left.grad(variable)
      .asInstanceOf[Expression[V, KL, Plus[M, CKL]]]
    val rightGrad = right.grad(variable)

    implicit val ckToInt = right.shape.dom.toInt
    implicit val ckrToInt = right.shape.mod.toInt
    val lor = (leftGrad outer right)
      .shiftLeft[CKR]
      .transpose[CKR, M]
      .shiftRight[M]
    val leftRes: Expression[V, Plus[KL, KR], Plus[M, Plus[CKR, CKL]]] =
      lor.asInstanceOf[Expression[V, Plus[KL, KR], Plus[M, Plus[CKR, CKL]]]]

    implicit val mpckr = new ToInt[Plus[M, CKR]] {
      override def apply() = implicitly[ToInt[M]].apply() + ckrToInt.apply()
    }
    val rightRes = (left outer rightGrad)
      .asInstanceOf[Expression[V, Plus[KL, KR], Plus[M, Plus[CKR, CKL]]]]

    leftRes + rightRes
  }

}

