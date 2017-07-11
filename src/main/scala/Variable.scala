import breeze.linalg._

sealed trait VariableLike[V] {

  def eval(context: Context): V = throw new NotImplementedError("No eval provided")

}

trait ScalarVariableLike extends VariableLike[Float] {

  import Variable._

  private val upstream = this

  def grad(scalar: ScalarVariableLike)(implicit model: Model): ScalarVariableLike = {
    if (scalar == this) {
      1.0f
    } else {
      0.0f
    }
  }

  def grad(vector: VectorVariableLike)(implicit model: Model): VectorVariableLike = {
    new VectorVariable(vector.length) {
      override def eval(context: Context) = DenseVector.zeros(vector.length)
    }
  }

  def unary_-(): ScalarVariableLike = {
    new ScalarVariable() {
      override def eval(context: Context) = {
        -context.eval(upstream)
      }

      override def grad(scalar: ScalarVariableLike)(implicit model: Model) = {
        -upstream.grad(scalar)
      }

      override def grad(vector: VectorVariableLike)(implicit model: Model) = {
        -upstream.grad(vector)
      }
    }
  }

  def +(other: ScalarVariableLike): ScalarVariableLike = {
    new ScalarVariable() {
      override def eval(context: Context) = {
        context.eval(upstream) + context.eval(other)
      }

      override def grad(scalar: ScalarVariableLike)(implicit model: Model) = {
        upstream.grad(scalar) + other.grad(scalar)
      }

      override def grad(vector: VectorVariableLike)(implicit model: Model) = {
        upstream.grad(vector) + other.grad(vector)
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
        upstream.grad(scalar) + other.grad(scalar)
      }

      override def grad(vector: VectorVariableLike)(implicit model: Model) = {
        upstream.grad(vector) + other.grad(vector)
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

  def grad(scalar: ScalarVariableLike)(implicit model: Model): VectorVariableLike = {
    new VectorVariable(length) {
      override def eval(context: Context) = DenseVector.zeros(length)
    }
  }

  def grad(vector: VectorVariableLike)(implicit model: Model): MatrixVariableLike = {
    if (this == vector) {
      new MatrixVariable(length, length) {
        override def eval(context: Context) = DenseMatrix.eye(length)
      }
    } else {
      new MatrixVariable(length, length) {
        override def eval(context: Context) = DenseMatrix.zeros(length, length)
      }
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
    new VectorVariable(length) {
      override def eval(context: Context) = {
        context.eval(upstream) *:* context.eval(other)
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
}

case class ScalarVariable() extends ScalarVariableLike

case class VectorVariable(length: Int) extends VectorVariableLike

case class MatrixVariable(rows: Int, cols: Int) extends MatrixVariableLike

object Variable {

  implicit def toScalar(value: Float)(implicit model: Model): ScalarVariable = {
    val variable = ScalarVariable()
    model.withConstant(variable, value)
    variable
  }

  implicit def toVector(value: DenseVector[Float])(implicit model: Model): VectorVariable = {
    val variable = VectorVariable(value.length)
    model.withConstant(variable, value)
    variable
  }

}
