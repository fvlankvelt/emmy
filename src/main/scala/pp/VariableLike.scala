package pp

import breeze.linalg.{DenseMatrix, DenseVector}

trait VariableLike[V, T <: VariableLike[V, T]] {

  import VariableLike._

  def repr: T

  def eval(context: Context): V = {
    throw new NotImplementedError("No eval provided")
  }

  def unary_-()(implicit op: NegOp[V, T]): T = op(repr)

  def +(other: T)(implicit op: AddOp[V, T]) = op(repr, other)

//  def +[W, O <: VariableLike[W, O]](other: O)(implicit op: AddOp[V, T], widen: Widen[O, T]) = op(repr, widen(other, repr))

  def -(other: T)(implicit op: SubOp[V, T]) = op(repr, other)

//  def -[W, O <: VariableLike[W, O]](other: O)(implicit op: SubOp[V, T], widen: Widen[O, T]) = op(repr, widen(other, repr))

  def *(other: T)(implicit op: MulOp[V, T]) = op(repr, other)

//  def *[W, O <: VariableLike[W, O]](other: O)(implicit op: MulOp[V, T], widen: Widen[O, T]) = op(repr, widen(other, repr))

  def /(other: T)(implicit op: DivOp[V, T]) = op(repr, other)

  def **(other: T)(implicit op: PowOp[V, T]) = op(repr, other)
}

object VariableLike {

  trait Widen[From, To] extends ((From, To) => To)

  implicit def toScalar(value: Float): ScalarVariable = {
    new ScalarVariable("const") {
      override def eval(context: Context) = value

      override def grad(scalar: ScalarVariableLike) = {
        if (scalar == this) {
          Some(toScalar(1.0f))
        } else {
          None
        }
      }

      override def grad(vector: VectorVariableLike) = {
        None
      }
    }
  }

  implicit def toVector(value: DenseVector[Float]): VectorVariable = {
    new VectorVariable(value.length) {
      override def eval(context: Context) = value

      override def grad(scalar: ScalarVariableLike) = {
        None
      }

      override def grad(vector: VectorVariableLike) = {
        if (vector == this) {
          val mat: DenseMatrix[Float] = DenseMatrix.eye(value.length)
          Some(VariableLike.toMatrix(mat))
        } else {
          None
        }
      }
    }
  }

  implicit def toMatrix(mat: DenseMatrix[Float]): MatrixVariable = {
    new MatrixVariable(mat.rows, mat.cols) {
      override def eval(context: Context) = mat

      override def grad(scalar: ScalarVariableLike) = None
    }
  }

  implicit object scalarToVector extends Widen[ScalarVariableLike, VectorVariableLike] {
    override def apply(scalar: ScalarVariableLike, hint: VectorVariableLike) = {
      scalar.toVector(hint.length)
    }
  }

  implicit object vectorToMatrix extends Widen[VectorVariableLike, MatrixVariableLike] {
    override def apply(scalar: VectorVariableLike, hint: MatrixVariableLike) = {
      scalar.toMatrix(hint.cols)
    }
  }

}
