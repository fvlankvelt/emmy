package pp.tensor

import breeze.math.Semiring

import scala.reflect.ClassTag

case class Variable[V: Semiring : ClassTag, L <: Nat : ToInt](dom: Domain[L])
  extends Expression[V, L, Zero] {

  val ringV = implicitly[Semiring[V]]
  val ctV = implicitly[ClassTag[V]]

  val shape = TensorShape(dom, Domain())

  override def eval() = {
    throw new NotImplementedError("Variable should not be evaluated")
  }

  override def grad[M <: Nat : ToInt](variable: Variable[V, M]): Expression[V, L, Plus[M, Nat._0]] = {
    if (this eq variable) {
      new ConstantExpression[V, L, L](Tensor.eye[V, L](dom)) // ugh, but cast will succeed
    } else {
      new ConstantExpression[V, L, M](Tensor[V, L, M](dom, variable.dom))
    }
  }.asInstanceOf[Expression[V, L, Plus[M, Nat._0]]]
}

