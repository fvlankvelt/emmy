package pp

trait VariableLike[V, T <: VariableLike[V, T]] {

  def repr: T

  def eval(context: Context): V = {
    throw new NotImplementedError("No eval provided")
  }

  def unary_-()(implicit op: NegOp[V, T]): T = op(repr)

  def +(other: T)(implicit op: AddOp[V, T]) = op(repr, other)
}
