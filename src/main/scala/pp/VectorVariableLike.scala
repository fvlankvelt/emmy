package pp

import breeze.linalg.{DenseMatrix, DenseVector}

trait VectorVariableLike extends VariableLike[DenseVector[Float], VectorVariableLike] {

  override def repr : VectorVariableLike = this

  val length: Int

  private val upstream = this

  def grad(scalar: ScalarVariableLike)(implicit model: Model): Option[VectorVariableLike] = None

  def grad(vector: VectorVariableLike)(implicit model: Model): Option[MatrixVariableLike] = {
    if (this == vector) {
      Some(new MatrixVariable(length, length) {
        override def eval(context: Context) = DenseMatrix.eye(length)
      })
    } else {
      None
    }
  }

  def toMatrix(cols: Int) = {
    new MatrixVariable(length, cols) {
      override def eval(context: Context) = {
        context.eval(upstream) * DenseVector.ones[Float](cols).t
      }
    }
  }


  def +(other: ScalarVariableLike) = {
    new VectorVariable(length) {

      override def eval(context: Context) = {
        val upstreamValue = context.eval(upstream)
        upstreamValue + context.eval(other)
      }
    }
  }

  def +(other: MatrixVariableLike): MatrixVariableLike = {
    assert(length == other.rows)

    new MatrixVariable(length, other.cols) {
      override def eval(context: Context) = {
        context.eval(other)(::, breeze.linalg.*) +:+ context.eval(upstream)
      }
    }
  }

  def -(other: VectorVariableLike) = {
    new VectorVariable(length) {
      override def eval(context: Context) = {
        context.eval(upstream) -:- context.eval(other)
      }
    }
  }

  def *(other: VectorVariableLike) = {
    assert (length == other.length)
    new VectorVariable(length) {
      override def eval(context: Context) = {
        context.eval(upstream) *:* context.eval(other)
      }

      override def grad(vector: VectorVariableLike)(implicit model: Model) = {
        val upGrad = upstream.grad(vector).map { upstreamMat =>
            new MatrixVariable(other.length, vector.length) {
              override def eval(context: Context) =
                context.eval(upstreamMat)(::, breeze.linalg.*) *:* context.eval(other)
            }
          }
        val otGrad = other.grad(vector).map { otherMat =>
            new MatrixVariable(other.length, vector.length) {
              override def eval(context: Context) =
                context.eval(otherMat)(::, breeze.linalg.*) *:* context.eval(upstream)
            }
          }
        (upGrad, otGrad) match {
          case (None, _) => otGrad
          case (_, None) => upGrad
          case (Some(upstreamMat), Some(otherMat)) => Some(
            new MatrixVariable(other.length, vector.length) {
              override def eval(context: Context) = {
                context.eval(upstreamMat) +:+ context.eval(otherMat)
              }
            }
          )
        }
      }
    }
  }

  def /(other: ScalarVariableLike) = {
    new VectorVariable(length) {
      override def eval(context: Context) = {
        context.eval(upstream) /:/ context.eval(other)
      }
    }
  }

  def **(other: ScalarVariableLike) = {
    new VectorVariable(length) {
      override def eval(context: Context) = {
        context.eval(upstream) ^:^ context.eval(other)
      }
    }
  }

}
