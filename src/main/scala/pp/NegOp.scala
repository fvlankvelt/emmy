package pp

import breeze.linalg.{DenseMatrix, DenseVector}

trait NegOp[V, T <: VariableLike[V, T]] extends (T => T)

object NegOp {

  implicit object MatrixNegOp extends NegOp[DenseMatrix[Float], MatrixVariableLike] {
    override def apply(vector: MatrixVariableLike) =
      new MatrixVariable(vector.rows, vector.cols) {
        override def eval(context: Context) = {
          -context.eval(vector)
        }
      }
  }

  implicit object VectorNegOp extends NegOp[DenseVector[Float], VectorVariableLike] {
    override def apply(vector: VectorVariableLike) =
      new VectorVariable(vector.length) {
        override def eval(context: Context) = {
          -context.eval(vector)
        }

        override def grad(scalar: ScalarVariableLike) = {
          vector.grad(scalar).map(-_)
        }

        override def grad(vector: VectorVariableLike) = {
          vector.grad(vector).map(-_)
        }
      }
  }

  implicit object ScalarNegOp extends NegOp[Float, ScalarVariableLike] {
    override def apply(scalar: ScalarVariableLike) =
      new ScalarVariable("-!") {
        override def eval(context: Context) = {
          val value = context.eval(scalar)
          -value
        }

        override def grad(scalar: ScalarVariableLike) = {
          scalar.grad(scalar).map(-_)
        }

        override def grad(vector: VectorVariableLike) = {
          scalar.grad(vector).map(-_)
        }
      }
  }

}
