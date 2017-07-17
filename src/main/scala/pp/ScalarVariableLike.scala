package pp

import breeze.linalg._

trait ScalarVariableLike extends VariableLike[Float, ScalarVariableLike] {

  override def repr: ScalarVariableLike = this

  def grad(scalar: ScalarVariableLike): Option[ScalarVariableLike] = {
    if (scalar == this) {
      Some(1.0f)
    } else {
      None
    }
  }

  def grad(vector: VectorVariableLike): Option[VectorVariableLike] = {
    None
  }

  def toVector(length: Int): VectorVariableLike = {
    val upstream = this
    new VectorVariable(length) {
      override def eval(context: Context) = {
        val value = context.eval(upstream)
        DenseVector.fill(length, value)
      }

      override def grad(scalar: ScalarVariableLike) = {
        upstream.grad(scalar).map(_.toVector(length))
      }

      override def grad(vector: VectorVariableLike) = {
        upstream.grad(vector).map(_.toMatrix(length).transpose)
      }

    }
  }

  def /(other: ScalarVariableLike): ScalarVariableLike = {
    val upstream = this
    new ScalarVariable("/") {
      override def eval(context: Context) = {
        context.eval(upstream) / context.eval(other)
      }

      override def grad(scalar: ScalarVariableLike) = {
        val upGrad = upstream.grad(scalar).map { g =>
          g / other
        }
        val otGrad = other.grad(scalar).map { g =>
          -upstream / (other * other)
        }
        (upGrad, otGrad) match {
          case (None, _) => otGrad
          case (_, None) => upGrad
          case _ => Some(upGrad.get + otGrad.get)
        }
      }
    }
  }

  def /(other: VectorVariableLike) = {
    val upstream = this
    new VectorVariable(other.length) {
      override def eval(context: Context) = {
        context.eval(upstream) /:/ context.eval(other)
      }
    }
  }

  def **(other: ScalarVariableLike): ScalarVariableLike = {
    val upstream = this
    new ScalarVariable("**") {
      override def eval(context: Context) = {
        Math.pow(context.eval(upstream), context.eval(other)).toFloat
      }

      override def grad(scalar: ScalarVariableLike) = {
        import Function._
        val upGrad = upstream.grad(scalar).map { g =>
          val exp = other - 1.0f
          g * other * (upstream ** exp)
        }
        val otGrad = other.grad(scalar).map { g =>
          g * log(upstream) * (upstream ** other)
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
