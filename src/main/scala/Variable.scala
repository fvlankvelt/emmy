import breeze.linalg._

sealed trait VariableLike[V] {

  def eval(context: Context): V = throw new NotImplementedError("No eval provided")

}

trait ScalarVariableLike extends VariableLike[Float] {

  private val upstream = this

  def unary_-() = {
    new ScalarVariable() {
      override def eval(context: Context) = - context.eval(upstream)
    }
  }

  def +(other: ScalarVariableLike) = {
    new ScalarVariable() {
      override def eval(context: Context) =
        context.eval(upstream) + context.eval(other)
    }
  }

  def +(other: VectorVariableLike) = {
    new VectorVariable(other.length) {
      override def eval(context: Context) = {
        val upstreamValue = context.eval(upstream)
        context.eval(other) + upstreamValue
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

case class ScalarVariable() extends ScalarVariableLike

case class VectorVariable(length: Int) extends VectorVariableLike

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
