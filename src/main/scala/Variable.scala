import breeze.linalg._

sealed trait VariableLike[V] {

  def eval(context: Context): V = throw new NotImplementedError("No eval provided")

}

case class ScalarVariable() extends VariableLike[Float] {

  private val upstream = this

  def unary_-() = {
    new ScalarVariable() {
      override def eval(context: Context) = - context.eval(upstream)
    }
  }

  def +(other: ScalarVariable) = {
    new ScalarVariable() {
      override def eval(context: Context) =
        context.eval(upstream) + context.eval(other)
    }
  }

  def +(other: VectorVariable) = {
    new VectorVariable(other.length) {
      override def eval(context: Context) = {
        val upstreamValue = context.eval(upstream)
        context.eval(other) + upstreamValue
      }
    }
  }

  def -(other: ScalarVariable) = {
    new ScalarVariable() {
      override def eval(context: Context) = {
        context.eval(upstream) - context.eval(other)
      }
    }
  }

  def *(other: ScalarVariable) = {
    new ScalarVariable() {
      override def eval(context: Context) =
        context.eval(upstream) * context.eval(other)
    }
  }

  def /(other: ScalarVariable) = {
    new ScalarVariable() {
      override def eval(context: Context) = {
        context.eval(upstream) / context.eval(other)
      }
    }
  }

  def /(other: VectorVariable) = {
    new VectorVariable(other.length) {
      override def eval(context: Context) = {
        context.eval(upstream) /:/ context.eval(other)
      }
    }
  }

  def **(other: ScalarVariable) = {
    new ScalarVariable() {
      override def eval(context: Context) = {
        Math.pow(context.eval(upstream), context.eval(other)).toFloat
      }
    }
  }

}

case class VectorVariable(length: Int) extends VariableLike[DenseVector[Float]] {

  private val upstream = this

  def unary_-() = {
    new VectorVariable(length) {
      override def eval(context: Context) = - context.eval(upstream)
    }
  }

  def +(other: ScalarVariable) = {
    new VectorVariable(length) {

      override def eval(context: Context) = {
        val upstreamValue = context.eval(upstream)
        upstreamValue + context.eval(other)
      }
    }
  }

  def +(other: VectorVariable) = {
    new VectorVariable(length) {
      override def eval(context: Context) = {
        context.eval(upstream) +:+ context.eval(other)
      }
    }
  }

  def -(other: VectorVariable) = {
    new VectorVariable(length) {
      override def eval(context: Context) = {
        context.eval(upstream) -:- context.eval(other)
      }
    }
  }

  def *(other: VectorVariable) = {
    new VectorVariable(length) {
      override def eval(context: Context) = {
        context.eval(upstream) *:* context.eval(other)
      }
    }
  }

  def **(other: ScalarVariable) = {
    new VectorVariable(length) {
      override def eval(context: Context) = {
        context.eval(upstream) ^:^ context.eval(other)
      }
    }
  }

}

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
