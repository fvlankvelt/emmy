package pp.tensor

case class ConstantExpression[K <: Nat, CK <: Nat]
(
  value: Tensor[K, CK]
) extends Expression[K, CK] {

  val shape = TensorShape(value.dom, value.mod)

  override def eval() = value

  override def grad[M <: Nat : ToInt](variable: Variable[M]) = {
    new ConstantExpression[K, Plus[M, CK]](Tensor[K, Plus[M, CK]](shape.dom, Domain.join(variable.dom, shape.mod)))
  }
}

