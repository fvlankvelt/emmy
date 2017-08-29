package pp.tensor

case class Variable[L <: Nat : ToInt](dom: Domain[L])
  extends Expression[L, Zero] {

  val shape = TensorShape(dom, Domain())

  override def eval() = {
    throw new NotImplementedError("Variable should not be evaluated")
  }

  override def grad[M <: Nat : ToInt](variable: Variable[M]): Expression[L, Plus[M, Nat._0]] = {
    if (this eq variable) {
      new ConstantExpression[L, L](Tensor.eye[L](dom)) // ugh, but cast will succeed
    } else {
      new ConstantExpression[L, M](Tensor[L, M](dom, variable.dom))
    }
  }.asInstanceOf[Expression[L, Plus[M, Nat._0]]]
}

