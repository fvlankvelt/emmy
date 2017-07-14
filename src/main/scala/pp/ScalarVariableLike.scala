package pp

import breeze.linalg._

trait ScalarVariableLike extends VariableLike[Float, ScalarVariableLike] {

  override def repr : ScalarVariableLike = this

  import Variable._

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

  def toVector(length: Int) = {
    val upstream = this
    new VectorVariable(length) {
      override def eval(context: Context) = {
        val value = context.eval(upstream)
        DenseVector.fill(length, value)
      }
    }
  }

  def +(other: VectorVariableLike): VectorVariableLike = {
    val upstream = this
    new VectorVariable(other.length) {
      override def eval(context: Context) = {
        val reprValue = context.eval(repr)
        context.eval(other) + reprValue
      }

      override def grad(scalar: ScalarVariableLike)(implicit model: Model) = {
        val upGrad = upstream.grad(scalar)
        val otGrad = other.grad(scalar)
        (upGrad, otGrad) match {
          case (None, _) => otGrad
          case (_, None) => upGrad.map { _.toVector(other.length) }
          case _ => Some(upGrad.get + otGrad.get)
        }
      }

      override def grad(vector: VectorVariableLike)(implicit model: Model) = {
        val upGrad = upstream.grad(vector)
        val otGrad = other.grad(vector)
        (upGrad, otGrad) match {
          case (None, _) => otGrad
          case (_, None) => upGrad.map { _.toMatrix(vector.length) }
          case _ => Some(upGrad.get + otGrad.get)
        }
      }
    }
  }

  def -(other: ScalarVariableLike) = {
    new ScalarVariable() {
      override def eval(context: Context) = {
        context.eval(repr) - context.eval(other)
      }
    }
  }

  def *(other: ScalarVariableLike) = {
    new ScalarVariable() {
      override def eval(context: Context) =
        context.eval(repr) * context.eval(other)
    }
  }

  def /(other: ScalarVariableLike) = {
    new ScalarVariable() {
      override def eval(context: Context) = {
        context.eval(repr) / context.eval(other)
      }
    }
  }

  def /(other: VectorVariableLike) = {
    new VectorVariable(other.length) {
      override def eval(context: Context) = {
        context.eval(repr) /:/ context.eval(other)
      }
    }
  }

  def **(other: ScalarVariableLike) = {
    new ScalarVariable() {
      override def eval(context: Context) = {
        Math.pow(context.eval(repr), context.eval(other)).toFloat
      }
    }
  }

}
