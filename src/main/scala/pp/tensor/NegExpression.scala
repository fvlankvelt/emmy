package pp.tensor

import breeze.math.Semiring

import scala.reflect.ClassTag

case class NegExpression[
V: ClassTag : Semiring,
K <: Nat,
CK <: Nat
](upstream: Expression[V, K, CK])
  extends Expression[V, K, CK] {

  val ringV = implicitly[Semiring[V]]
  val ctV = implicitly[ClassTag[V]]

  override val shape = upstream.shape

  override def eval() = {
    new Tensor[V, K, CK](shape.dom, shape.mod, -upstream.eval().data)
  }

  override def grad[M <: Nat : ToInt](variable: Variable[V, M]) = {
    -upstream.grad(variable)
  }
}

