package pp.tensor

import breeze.linalg.DenseMatrix
import breeze.math.Field

import scala.reflect.ClassTag

object Function {

  trait ValueConverter[V] {
    def toDouble(value: V): Double
    def toV(value: Double): V
    def wrap(fn: Double => Double) : V => V =
      (value: V) => toV(fn(toDouble(value)))
  }
  implicit val doubleConverter : ValueConverter[Double] = new ValueConverter[Double] {
    override def toDouble(value: Double) = value
    override def toV(value: Double) = value
  }
  implicit val floatConverter : ValueConverter[Float] = new ValueConverter[Float] {
    override def toDouble(value: Float) = value
    override def toV(value: Double) = value.toFloat
  }

  private implicit def logImpl[V](implicit converter: ValueConverter[V]) = new breeze.numerics.log.Impl[V, V] {
    override def apply(v: V) = converter.wrap(breeze.numerics.log.logDoubleImpl.apply)(v)
  }

  def log[V: ClassTag : Field : ValueConverter, K <: Nat, CK <: Nat](variable: Expression[V, K, CK]): Expression[V, K, CK] =
    new Expression[V, K, CK] {

      val ringV = variable.ringV
      val ctV = variable.ctV
      val shape = variable.shape

      override def eval() = {
        val upstream = variable.eval()
        Tensor[V, K, CK](shape.dom, shape.mod, breeze.numerics.log(upstream.data))
      }

      override def grad[M <: Nat : ToInt](other: Variable[V, M]) = {
        val gradient = variable.grad(other)
        gradient / variable.broadcastCov(other.dom)
      }
    }

  private implicit def lgammaImpl[V](implicit converter: ValueConverter[V]) = new breeze.numerics.lgamma.Impl[V, V] {
    override def apply(v: V) = converter.wrap(breeze.numerics.lgamma.lgammaImplDouble.apply)(v)
  }

  def lgamma[V: ClassTag : Field : ValueConverter, K <: Nat, CK <: Nat](variable: Expression[V, K, CK]): Expression[V, K, CK] =
    new Expression[V, K, CK] {

      val ringV = variable.ringV
      val ctV = variable.ctV
      val shape = variable.shape

      override def eval() = {
        val upstream = variable.eval()
        Tensor[V, K, CK](shape.dom, shape.mod, breeze.numerics.lgamma(upstream.data))
      }

      override def grad[M <: Nat : ToInt](other: Variable[V, M]) = {
        val gradient = variable.grad(other)
        gradient * digamma(variable).broadcastCov(other.dom)
      }
    }

  private implicit def digammaImpl[V](implicit converter: ValueConverter[V]) = new breeze.numerics.digamma.Impl[V, V] {
    override def apply(v: V) = converter.wrap(breeze.numerics.digamma.digammaImplDouble.apply)(v)
  }

  def digamma[V: ClassTag : Field : ValueConverter, K <: Nat, CK <: Nat](variable: Expression[V, K, CK]): Expression[V, K, CK] =
    new Expression[V, K, CK] {

      val ringV = variable.ringV
      val ctV = variable.ctV
      val shape = variable.shape

      override def eval() = {
        val upstream = variable.eval()
        Tensor[V, K, CK](shape.dom, shape.mod, breeze.numerics.digamma(upstream.data))
      }

      override def grad[M <: Nat : ToInt](other: Variable[V, M]) =
        throw new NotImplementedError()
   }

  def sum[V : ClassTag : Field, K <: Nat, CK <: Nat](expr: Expression[V, K, CK]): Expression[V, Zero, Zero] = {
    new Expression[V, Zero, Zero] {
      val ringV = expr.ringV
      val ctV = expr.ctV

      override def shape = TensorShape(Domain(), Domain())

      override def eval() = {
        val s = breeze.linalg.sum(expr.eval().data)
        Tensor[V, Zero, Zero](shape.dom, shape.mod, DenseMatrix.create(1, 1, Array(s)))
      }

      override def grad[M <: Nat : ToInt](variable: Variable[V, M]) = ???
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
