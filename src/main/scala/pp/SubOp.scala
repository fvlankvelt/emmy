package pp

import breeze.linalg.{DenseMatrix, DenseVector}

trait SubOp[V, T <: VariableLike[V, T]] extends ((T, T) => T)

object SubOp {

  implicit object MatrixSubOp extends SubOp[DenseMatrix[Float], MatrixVariableLike] {
    override def apply(left: MatrixVariableLike, right: MatrixVariableLike) = {
      assert(left.rows == right.rows && left.cols == right.cols)
      new MatrixVariable(left.rows, left.cols) {
        override def eval(context: Context) = {
          context.eval(left) -:- context.eval(right)
        }
      }
    }
  }

  implicit object VectorSubOp extends SubOp[DenseVector[Float], VectorVariableLike] {
    override def apply(self: VectorVariableLike, other: VectorVariableLike) = {
      assert(self.length == other.length)
      new VectorVariable(self.length) {
        override def eval(context: Context) = {
          context.eval(self) -:- context.eval(other)
        }

        override def grad(scalar: ScalarVariableLike) = {
          val upGrad = self.grad(scalar)
          val otGrad = other.grad(scalar).map(-_)
          (upGrad, otGrad) match {
            case (None, _) => otGrad
            case (_, None) => upGrad
            case _ => Some(upGrad.get + otGrad.get)
          }
        }

        override def grad(vector: VectorVariableLike) = {
          val upGrad = self.grad(vector)
          val otGrad = other.grad(vector).map(-_)
          (upGrad, otGrad) match {
            case (None, _) => otGrad
            case (_, None) => upGrad
            case _ => Some(upGrad.get + otGrad.get)
          }
        }
      }
    }
  }

  implicit object ScalarSubOp extends SubOp[Float, ScalarVariableLike] {
    override def apply(left: ScalarVariableLike, right: ScalarVariableLike) =
      new ScalarVariable("-") {
        override def eval(context: Context) = {
          context.eval(left) - context.eval(right)
        }

        override def grad(scalar: ScalarVariableLike) = {
          val upGrad = left.grad(scalar)
          val otGrad = right.grad(scalar).map(-_)
          (upGrad, otGrad) match {
            case (None, _) => otGrad
            case (_, None) => upGrad
            case _ => Some(upGrad.get + otGrad.get)
          }
        }

        override def grad(vector: VectorVariableLike) = {
          val upGrad = left.grad(vector)
          val otGrad = right.grad(vector).map(-_)
          (upGrad, otGrad) match {
            case (None, _) => otGrad
            case (_, None) => upGrad
            case _ => Some(upGrad.get + otGrad.get)
          }
        }
      }
  }

}
