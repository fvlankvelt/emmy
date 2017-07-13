import breeze.linalg.Axis

object Function {

  def log(variable: VectorVariableLike): VectorVariable =
    new VectorVariable(variable.length) {
      override def eval(context: Context) = {
        val upstream = context.eval(variable)
        breeze.numerics.log(upstream)
      }
    }

  def log(variable: ScalarVariableLike): ScalarVariable =
    new ScalarVariable() {
      override def eval(context: Context) = {
        val upstream = context.eval(variable)
        breeze.numerics.log(upstream)
      }
    }

  def sum(variable: VectorVariableLike): ScalarVariable =
    new ScalarVariable {
      override def eval(context: Context) = {
        val upstream = context.eval(variable)
        breeze.linalg.sum(upstream)
      }

      override def grad(vector: VectorVariableLike)(implicit model: Model) = {
        variable.grad(vector).map { mat =>
          new VectorVariable(variable.length) {
            override def eval(context: Context) = {
              val matVal = context.eval(mat)
              breeze.linalg.sum(matVal, Axis._1)
            }
          }
        }
      }
    }
}
