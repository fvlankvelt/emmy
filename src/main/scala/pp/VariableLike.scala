package pp

import breeze.linalg.DenseVector

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
    }
  }

  implicit def toVector(value: DenseVector[Float]): VectorVariable = {
    new VectorVariable(value.length) {
      override def eval(context: Context) = value
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
