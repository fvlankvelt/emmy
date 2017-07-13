import breeze.linalg._

sealed trait VariableLike[V] {

  def eval(context: Context): V = {
    throw new NotImplementedError("No eval provided")
  }

}

trait ScalarVariableLike extends VariableLike[Float] {

  import Variable._

  private val upstream = this

  def grad(scalar: ScalarVariableLike)(implicit model: Model): Option[ScalarVariableLike] = {
    if (scalar == this) {
      Some(1.0f)
    } else {
      None
    }
  }

  def grad(vector: VectorVariableLike)(implicit model: Model): Option[VectorVariableLike] = {
    None
  }

  def unary_-(): ScalarVariableLike = {
    new ScalarVariable() {
      override def eval(context: Context) = {
        -context.eval(upstream)
      }

      override def grad(scalar: ScalarVariableLike)(implicit model: Model) = {
        upstream.grad(scalar).map(-_)
      }

      override def grad(vector: VectorVariableLike)(implicit model: Model) = {
        upstream.grad(vector).map(-_)
      }
    }
  }

  def +(other: ScalarVariableLike): ScalarVariableLike = {
    new ScalarVariable() {
      override def eval(context: Context) = {
        context.eval(upstream) + context.eval(other)
      }

      override def grad(scalar: ScalarVariableLike)(implicit model: Model) = {
        val upGrad = upstream.grad(scalar)
        val otGrad = other.grad(scalar)
        (upGrad, otGrad) match {
          case (None, _) => otGrad
          case (_, None) => upGrad
          case _ => Some(otGrad.get + upGrad.get)
        }
      }

      override def grad(vector: VectorVariableLike)(implicit model: Model) = {
        val upGrad = upstream.grad(vector)
        val otGrad = other.grad(vector)
        (upGrad, otGrad) match {
          case (None, _) => otGrad
          case (_, None) => upGrad
          case _ => Some(otGrad.get + upGrad.get)
        }
      }
    }
  }

  def +(other: VectorVariableLike): VectorVariableLike = {
    new VectorVariable(other.length) {
      override def eval(context: Context) = {
        val upstreamValue = context.eval(upstream)
        context.eval(other) + upstreamValue
      }

      override def grad(scalar: ScalarVariableLike)(implicit model: Model) = {
        val upGrad = upstream.grad(scalar)
        val otGrad = other.grad(scalar)
        (upGrad, otGrad) match {
          case (None, _) => otGrad
          case (_, None) => upGrad.map { g =>
            new VectorVariable(other.length) {
              override def eval(context: Context) = {
                val value = context.eval(g)
                DenseVector.fill(other.length, value)
              }
            }
          }
          case _ => Some(upGrad.get + otGrad.get)
        }
      }

      override def grad(vector: VectorVariableLike)(implicit model: Model) = {
        val upGrad = upstream.grad(vector)
        val otGrad = other.grad(vector)
        (upGrad, otGrad) match {
          case (None, _) => otGrad
          case (_, None) => upGrad.map { g =>
            new MatrixVariable(other.length, vector.length) {
              override def eval(context: Context) = {
                context.eval(g) * DenseVector.ones[Float](vector.length).t
              }
            }
          }
          case _ => Some(upGrad.get + otGrad.get)
        }
      }
    }
  }

  def -(other: ScalarVariableLike) = {
    new ScalarVariable() {
      override def eval(context: Context) = {
        context.eval(upstream) - context.eval(other)
      }
    }
  }

  def *(other: ScalarVariableLike) = {
    new ScalarVariable() {
      override def eval(context: Context) =
        context.eval(upstream) * context.eval(other)
    }
  }

  def /(other: ScalarVariableLike) = {
    new ScalarVariable() {
      override def eval(context: Context) = {
        context.eval(upstream) / context.eval(other)
      }
    }
  }

  def /(other: VectorVariableLike) = {
    new VectorVariable(other.length) {
      override def eval(context: Context) = {
        context.eval(upstream) /:/ context.eval(other)
      }
    }
  }

  def **(other: ScalarVariableLike) = {
    new ScalarVariable() {
      override def eval(context: Context) = {
        Math.pow(context.eval(upstream), context.eval(other)).toFloat
      }
    }
  }

}

trait VectorVariableLike extends VariableLike[DenseVector[Float]] {

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

  def unary_-() = {
    new VectorVariable(length) {
      override def eval(context: Context) = - context.eval(upstream)
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

  def +(other: VectorVariableLike) = {
    new VectorVariable(length) {
      override def eval(context: Context) = {
        context.eval(upstream) +:+ context.eval(other)
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

trait MatrixVariableLike extends VariableLike[DenseMatrix[Float]] {
  val rows: Int
  val cols: Int

  private val upstream = this

  def +(other: MatrixVariableLike) = {
    assert(rows == other.rows && cols == other.cols)

    new MatrixVariable(rows, cols) {
      override def eval(context: Context) = {
        context.eval(upstream) + context.eval(other)
      }
    }
  }

  def mvp(vector: VectorVariableLike) = {
    assert(cols == vector.length)
    new VectorVariable(rows) {
      override def eval(context: Context) = {
        context.eval(upstream) * context.eval(vector)
      }
    }
  }
}

case class ScalarVariable() extends ScalarVariableLike

case class VectorVariable(length: Int) extends VectorVariableLike

case class MatrixVariable(rows: Int, cols: Int) extends MatrixVariableLike

object Variable {

  implicit def toScalar(value: Float)(implicit model: Model): ScalarVariable = {
    new ScalarVariable {
      override def eval(context: Context) = value
    }
  }

  implicit def toVector(value: DenseVector[Float])(implicit model: Model): VectorVariable = {
    new VectorVariable(value.length) {
      override def eval(context: Context) = value
    }
  }

}
