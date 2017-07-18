package pp

import breeze.linalg.Axis

object Function {

  def log(variable: ScalarVariableLike): ScalarVariable =
    new ScalarVariable("log") {
      override def eval(context: Context) = {
        val upstream = context.eval(variable)
        breeze.numerics.log(upstream)
      }

      override def grad(scalar: ScalarVariableLike) = {
        variable.grad(scalar).map { _ / variable }
      }

      override def grad(vector: VectorVariableLike) = {
        variable.grad(vector).map { _ / variable.toVector(vector.length) }
      }
    }

  def log(variable: VectorVariableLike): VectorVariable =
    new VectorVariable(variable.length) {
      override def eval(context: Context) = {
        val upstream = context.eval(variable)
        breeze.numerics.log(upstream)
      }

      override def grad(scalar: ScalarVariableLike) = {
        variable.grad(scalar).map { _ / variable }
      }

      override def grad(vector: VectorVariableLike) = {
        variable.grad(vector).map { _ / variable.toMatrix(vector.length) }
      }
    }

  def lgamma(variable: ScalarVariableLike): ScalarVariable =
    new ScalarVariable("lgamma") {
      override def eval(context: Context) = {
        val upstream = context.eval(variable)
        breeze.numerics.lgamma(upstream.toDouble).toFloat
      }

      override def grad(scalar: ScalarVariableLike) = {
        variable.grad(scalar).map { _ * digamma(variable) }
      }

      override def grad(vector: VectorVariableLike) = {
        variable.grad(vector).map { _ * digamma(variable).toVector(vector.length) }
      }
    }

  def digamma(variable: ScalarVariableLike): ScalarVariable =
    new ScalarVariable("digamma") {
      override def eval(context: Context) = {
        val upstream = context.eval(variable)
        breeze.numerics.digamma(upstream.toDouble).toFloat
      }

      override def grad(scalar: ScalarVariableLike) = {
        throw new NotImplementedError()
      }

      override def grad(vector: VectorVariableLike) = {
        throw new NotImplementedError()
      }
    }


  def sum(variable: VectorVariableLike): ScalarVariable =
    new ScalarVariable("sum") {
      override def eval(context: Context) = {
        val upstream = context.eval(variable)
        breeze.linalg.sum(upstream)
      }

      override def grad(scalar: ScalarVariableLike) = {
        variable.grad(scalar).map { sum }
      }

      override def grad(vector: VectorVariableLike) = {
        variable.grad(vector).map { mat => sum(mat.transpose) }
      }
    }

  def sum(variable: MatrixVariableLike): VectorVariableLike =
    new VectorVariable(variable.rows) {
      override def eval(context: Context) = {
        val matVal = context.eval(variable)
        breeze.linalg.sum(matVal, Axis._1)
      }
    }
}
