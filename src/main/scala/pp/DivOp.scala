package pp

import breeze.linalg.{DenseMatrix, DenseVector}

trait DivOp[V, T <: VariableLike[V, T]] extends ((T, T) => T)

object DivOp {

  implicit object MatrixDivOp extends DivOp[DenseMatrix[Float], MatrixVariableLike] {
    override def apply(upstream: MatrixVariableLike, other: MatrixVariableLike): MatrixVariableLike = {
      new MatrixVariable(upstream.rows, upstream.cols) {
        override def eval(context: Context) = {
          context.eval(upstream) /:/ context.eval(other)
        }

        override def grad(scalar: ScalarVariableLike) = {
          throw new NotImplementedError()
        }
      }
    }
  }

  implicit object VectorDivOp extends DivOp[DenseVector[Float], VectorVariableLike] {
    override def apply(upstream: VectorVariableLike, other: VectorVariableLike): VectorVariableLike = {
      new VectorVariable(upstream.length) {
        override def eval(context: Context) = {
          context.eval(upstream) /:/ context.eval(other)
        }

        override def grad(scalar: ScalarVariableLike) = {
          val upGrad = upstream.grad(scalar).map { _ / other }
          val otGrad = other.grad(scalar).map { g => -upstream * g / (other * other) }
          (upGrad, otGrad) match {
            case (None, _) => otGrad
            case (_, None) => upGrad
            case _ => Some(upGrad.get + otGrad.get)
          }
        }

        override def grad(vector: VectorVariableLike) = {
          val upGrad = upstream.grad(vector).map { _ / other.toMatrix(vector.length) }
          val otGrad = other.grad(vector).map { g =>
            g * (-upstream / (other * other)).toMatrix(vector.length)
          }
          (upGrad, otGrad) match {
            case (None, _) => otGrad
            case (_, None) => upGrad
            case _ => Some(upGrad.get + otGrad.get)
          }
        }
      }
    }
  }

  implicit object ScalarDivOp extends DivOp[Float, ScalarVariableLike] {
    override def apply(upstream: ScalarVariableLike, other: ScalarVariableLike): ScalarVariableLike = {
      new ScalarVariable("/") {
        override def eval(context: Context) = {
          context.eval(upstream) / context.eval(other)
        }

        override def grad(scalar: ScalarVariableLike) = {
          val upGrad = upstream.grad(scalar).map { _ / other }
          val otGrad = other.grad(scalar).map { g => -upstream * g / (other * other) }
          (upGrad, otGrad) match {
            case (None, _) => otGrad
            case (_, None) => upGrad
            case _ => Some(upGrad.get + otGrad.get)
          }
        }

        override def grad(vector: VectorVariableLike) = {
          val upGrad = upstream.grad(vector).map { _ / other.toVector(vector.length) }
          val otGrad = other.grad(vector).map { g =>
            val denom = (other * other).toVector(vector.length)
            -(g * upstream.toVector(vector.length)) / denom
          }
          (upGrad, otGrad) match {
            case (None, _) => otGrad
            case (_, None) => upGrad
            case _ => Some(upGrad.get + otGrad.get)
          }
        }
      }
    }
  }

}
