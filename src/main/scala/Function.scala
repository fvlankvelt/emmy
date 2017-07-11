object Function {

  def log(variable: VectorVariable): VectorVariable =
    new VectorVariable(variable.length) {
      override def eval(context: Context) = {
        val upstream = context.eval(variable)
        breeze.numerics.log(upstream)
      }
    }

  def log(variable: ScalarVariable): ScalarVariable =
    new ScalarVariable() {
      override def eval(context: Context) = {
        val upstream = context.eval(variable)
        breeze.numerics.log(upstream)
      }
    }

  def sum(variable: VectorVariable): ScalarVariable =
    new ScalarVariable {
      override def eval(context: Context) = {
        val upstream = context.eval(variable)
        breeze.linalg.sum(upstream)
      }
    }
}
