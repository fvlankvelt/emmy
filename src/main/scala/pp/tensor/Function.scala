package pp.tensor

import breeze.math.Field

import scala.reflect.ClassTag

object Function {

  def log[V: ClassTag : Field, K <: Nat, CK <: Nat](variable: Expression[V, K, CK]): Expression[V, K, CK] =
    new Expression[V, K, CK] {

      val ringV = variable.ringV
      val ctV = variable.ctV
      val shape = variable.shape

      override def eval() = {
        val upstream = variable.eval()
        Tensor[V, K, CK](shape.dom, shape.mod, breeze.numerics.log(upstream))
      }

      override def grad[M <: Nat : ToInt](other: Variable[V, M]) = {
        val gradient = variable.grad(other)
        gradient / variable.broadcastCov(other.dom)
      }
    }

  def lgamma[V: ClassTag : Field, K <: Nat, CK <: Nat](variable: Expression[V, K, CK]): Expression[V, K, CK] =
    new Expression[V, K, CK] {

      val ringV = variable.ringV
      val ctV = variable.ctV
      val shape = variable.shape

      override def eval() = {
        val upstream = variable.eval()
        Tensor[V, K, CK](shape.dom, shape.mod, breeze.numerics.lgamma(upstream))
      }

      override def grad[M <: Nat : ToInt](other: Variable[V, M]) = {
        val gradient = variable.grad(other)
        gradient * digamma(variable).broadcastCov(other.dom)
      }
    }

  def digamma[V: ClassTag : Field, K <: Nat, CK <: Nat](variable: Expression[V, K, CK]): Expression[V, K, CK] =
    new Expression[V, K, CK] {

      val ringV = variable.ringV
      val ctV = variable.ctV
      val shape = variable.shape

      override def eval() = {
        val upstream = variable.eval()
        Tensor[V, K, CK](shape.dom, shape.mod, breeze.numerics.digamma(upstream))
      }

      override def grad[M <: Nat : ToInt](other: Variable[V, M]) =
        throw new NotImplementedError()
   }

  /*

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

      override def grad(scalar: ScalarVariableLike) = {
        throw new NotImplementedError()
      }

      override def grad(vector: VectorVariableLike) = {
        throw new NotImplementedError()
      }
    }
  */
}
