package pp

import breeze.linalg._

trait ScalarVariableLike extends VariableLike[Float, ScalarVariableLike] {

  override def repr : ScalarVariableLike = this


  import Variable._

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
        val reprValue = context.eval(upstream)
        context.eval(other) + reprValue
      }

      override def grad(scalar: ScalarVariableLike) = {
        val upGrad = upstream.grad(scalar)
        val otGrad = other.grad(scalar)
        (upGrad, otGrad) match {
          case (None, _) => otGrad
          case (_, None) => upGrad.map { _.toVector(other.length) }
          case _ => Some(upGrad.get + otGrad.get)
        }
      }

      override def grad(vector: VectorVariableLike) = {
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

  def *(other: ScalarVariableLike): ScalarVariableLike = {
    val upstream = this
    new ScalarVariable("*") {
      override def eval(context: Context) =
        context.eval(upstream) * context.eval(other)

      override def grad(scalar: ScalarVariableLike) = {
        val upGrad = upstream.grad(scalar).map { g =>
          g * other
        }
        val otGrad = other.grad(scalar).map { g =>
          upstream * g
        }
        (upGrad, otGrad) match {
          case (None, _) => otGrad
          case (_, None) => upGrad
          case _ => Some(upGrad.get + otGrad.get)
        }
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
          g * other * (upstream ** (other - 1.0f))
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
