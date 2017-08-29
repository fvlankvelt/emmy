package pp.tensor

import breeze.linalg.DenseMatrix

object Function {

  def log[K <: Nat, CK <: Nat](variable: Expression[K, CK]): Expression[K, CK] =
    new Expression[K, CK] {

      val shape = variable.shape

      override def eval() = {
        val upstream = variable.eval()
        Tensor[K, CK](shape.dom, shape.mod, breeze.numerics.log(upstream.data))
      }

      override def grad[M <: Nat : ToInt](other: Variable[M]) = {
        val gradient = variable.grad(other)
        gradient / variable.broadcastCov(other.dom)
      }
    }

  def lgamma[K <: Nat, CK <: Nat](variable: Expression[K, CK]): Expression[K, CK] =
    new Expression[K, CK] {

      val shape = variable.shape

      implicit val lgammaImpl = new breeze.numerics.lgamma.Impl[Float, Float] {
        override def apply(v: Float) = breeze.numerics.lgamma.lgammaImplDouble(v.toDouble).toFloat
      }

      override def eval() = {
        val upstream = variable.eval()
        Tensor[K, CK](shape.dom, shape.mod, breeze.numerics.lgamma(upstream.data))
      }

      override def grad[M <: Nat : ToInt](other: Variable[M]) = {
        val gradient = variable.grad(other)
        gradient * digamma(variable).broadcastCov(other.dom)
      }
    }

  def digamma[K <: Nat, CK <: Nat](variable: Expression[K, CK]): Expression[K, CK] =
    new Expression[K, CK] {

      val shape = variable.shape

      implicit val digammaImpl = new breeze.numerics.digamma.Impl[Float, Float] {
        override def apply(v: Float) = breeze.numerics.digamma.digammaImplDouble(v.toDouble).toFloat
      }

      override def eval() = {
        val upstream = variable.eval()
        Tensor[K, CK](shape.dom, shape.mod, breeze.numerics.digamma(upstream.data))
      }

      override def grad[M <: Nat : ToInt](other: Variable[M]) =
        throw new NotImplementedError()
   }

  def sum[K <: Nat, CK <: Nat](expr: Expression[K, CK]): Expression[Zero, Zero] = {
    new Expression[Zero, Zero] {

      override def shape = TensorShape(Domain(), Domain())

      override def eval() = {
        val s = breeze.linalg.sum(expr.eval().data)
        Tensor[Zero, Zero](shape.dom, shape.mod, DenseMatrix.create[Float](1, 1, Array(s)))
      }

      override def grad[M <: Nat : ToInt](variable: Variable[M]) = ???
    }
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
