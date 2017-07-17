package pp

import breeze.linalg.{DenseMatrix, DenseVector}

trait AddOp[V, T <: VariableLike[V, T]] extends ((T, T) => T)

object AddOp {

  implicit object MatrixAddOp extends AddOp[DenseMatrix[Float], MatrixVariableLike] {
    override def apply(left: MatrixVariableLike, right: MatrixVariableLike) = {
      assert(left.rows == right.rows && left.cols == right.cols)
      new MatrixVariable(left.rows, left.cols) {
        override def eval(context: Context) = {
          context.eval(left) +:+ context.eval(right)
        }

        override def grad(scalar: ScalarVariableLike) = {
          val upGrad = left.grad(scalar)
          val otGrad = right.grad(scalar)
          (upGrad, otGrad) match {
            case (None, _) => otGrad
            case (_, None) => upGrad
            case _ => Some(otGrad.get + upGrad.get)
          }
        }
      }
    }
  }

  implicit object VectorAddOp extends AddOp[DenseVector[Float], VectorVariableLike] {
    override def apply(self: VectorVariableLike, other: VectorVariableLike) = {
      assert(self.length == other.length)
      new VectorVariable(self.length) {
        override def eval(context: Context) = {
          context.eval(self) +:+ context.eval(other)
        }

        override def grad(scalar: ScalarVariableLike) = {
          val upGrad = self.grad(scalar)
          val otGrad = other.grad(scalar)
          (upGrad, otGrad) match {
            case (None, _) => otGrad
            case (_, None) => upGrad
            case _ => Some(otGrad.get + upGrad.get)
          }
        }

        override def grad(vector: VectorVariableLike) = {
          val upGrad = self.grad(vector)
          val otGrad = other.grad(vector)
          (upGrad, otGrad) match {
            case (None, _) => otGrad
            case (_, None) => upGrad
            case _ => Some(otGrad.get + upGrad.get)
          }
        }
      }
    }
  }

  implicit object ScalarAddOp extends AddOp[Float, ScalarVariableLike] {
    override def apply(left: ScalarVariableLike, right: ScalarVariableLike) =
      new ScalarVariable("+") {
        override def eval(context: Context) = {
          context.eval(left) + context.eval(right)
        }

        override def grad(scalar: ScalarVariableLike) = {
          val upGrad = left.grad(scalar)
          val otGrad = right.grad(scalar)
          (upGrad, otGrad) match {
            case (None, _) => otGrad
            case (_, None) => upGrad
            case _ => Some(otGrad.get + upGrad.get)
          }
        }

        override def grad(vector: VectorVariableLike) = {
          val upGrad = left.grad(vector)
          val otGrad = right.grad(vector)
          (upGrad, otGrad) match {
            case (None, _) => otGrad
            case (_, None) => upGrad
            case _ => Some(otGrad.get + upGrad.get)
          }
        }

      }
  }

}
