package pp

import breeze.linalg.{DenseMatrix, DenseVector}

trait TimesOp[V, T <: VariableLike[V, T]] extends ((T, T) => T)

object TimesOp {

  implicit object MatrixTimesOp extends TimesOp[DenseMatrix[Float], MatrixVariableLike] {
    override def apply(left: MatrixVariableLike, right: MatrixVariableLike) = {
      assert(left.rows == right.rows && left.cols == right.cols)
      new MatrixVariable(left.rows, left.cols) {
        override def eval(context: Context) = {
          context.eval(left) *:* context.eval(right)
        }

        override def grad(scalar: ScalarVariableLike) = {
          val upGrad = left.grad(scalar).map { _ * right }
          val otGrad = right.grad(scalar).map { _ * left }
          (upGrad, otGrad) match {
            case (None, _) => otGrad
            case (_, None) => upGrad
            case _ => Some(upGrad.get + otGrad.get)
          }
        }
      }
    }
  }

  implicit object VectorTimesOp extends TimesOp[DenseVector[Float], VectorVariableLike] {
    override def apply(self: VectorVariableLike, other: VectorVariableLike) = {
      assert(self.length == other.length)
      new VectorVariable(self.length) {
        override def eval(context: Context) = {
          context.eval(self) *:* context.eval(other)
        }

        override def grad(scalar: ScalarVariableLike) = {
          val upGrad = self.grad(scalar).map { _ * other }
          val otGrad = other.grad(scalar).map { _ * self }
          (upGrad, otGrad) match {
            case (None, _) => otGrad
            case (_, None) => upGrad
            case _ => Some(upGrad.get + otGrad.get)
          }
        }

        override def grad(vector: VectorVariableLike) = {
          val upGrad = self.grad(vector).map { _ * other.toMatrix(vector.length) }
          val otGrad = other.grad(vector).map { _ * self.toMatrix(vector.length) }
          (upGrad, otGrad) match {
            case (None, _) => otGrad
            case (_, None) => upGrad
            case _ => Some(upGrad.get + otGrad.get)
          }
        }
      }
    }
  }

  implicit object ScalarTimesOp extends TimesOp[Float, ScalarVariableLike] {
    override def apply(left: ScalarVariableLike, right: ScalarVariableLike) =
      new ScalarVariable("*") {
        override def eval(context: Context) =
          context.eval(left) * context.eval(right)

        override def grad(scalar: ScalarVariableLike) = {
          val upGrad = left.grad(scalar).map { _ * right }
          val otGrad = right.grad(scalar).map { _ * left }
          (upGrad, otGrad) match {
            case (None, _) => otGrad
            case (_, None) => upGrad
            case _ => Some(upGrad.get + otGrad.get)
          }
        }

        override def grad(vector: VectorVariableLike) = {
          val upGrad = left.grad(vector).map { _ * right.toVector(vector.length) }
          val otGrad = right.grad(vector).map { _ * left.toVector(vector.length) }
          (upGrad, otGrad) match {
            case (None, _) => otGrad
            case (_, None) => upGrad
            case _ => Some(otGrad.get + upGrad.get)
          }
        }
      }
  }

}
