package pp

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
    new ScalarVariable("log") {
      override def eval(context: Context) = {
        val upstream = context.eval(variable)
        breeze.numerics.log(upstream)
      }

      override def grad(scalar: ScalarVariableLike) = {
        variable.grad(scalar).map { upGrad =>
          upGrad / variable
        }
      }
    }

  def lgamma(variable: ScalarVariableLike): ScalarVariable =
    new ScalarVariable("lgamma") {
      override def eval(context: Context) = {
        val upstream = context.eval(variable)
        breeze.numerics.lgamma(upstream.toDouble).toFloat
      }

      override def grad(scalar: ScalarVariableLike) = {
        variable.grad(scalar).map { upGrad =>
          upGrad * digamma(variable)
        }
      }
    }

  def digamma(variable: ScalarVariableLike): ScalarVariable =
    new ScalarVariable("digamma") {
      override def eval(context: Context) = {
        val upstream = context.eval(variable)
        breeze.numerics.digamma(upstream.toDouble).toFloat
      }
    }


  def sum(variable: VectorVariableLike): ScalarVariable =
    new ScalarVariable("sum") {
      override def eval(context: Context) = {
        val upstream = context.eval(variable)
        breeze.linalg.sum(upstream)
      }

      override def grad(vector: VectorVariableLike) = {
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
