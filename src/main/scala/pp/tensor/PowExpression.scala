package pp.tensor

import breeze.math.Field

import scala.reflect.ClassTag

case class PowExpression[
V: ClassTag : Field,
K <: Nat,
CK <: Nat
](base: Expression[V, K, CK], power: Expression[V, K, CK])
  extends Expression[V, K, CK] {

  assert(base.shape == power.shape)

  val ringV = implicitly[Field[V]]
  val ctV = implicitly[ClassTag[V]]

  override val shape = base.shape

  override def eval() = {
    val leftTensor = base.eval()
    val rightTensor = power.eval()
    Tensor[V, K, CK](shape.dom, shape.mod, leftTensor.data ^:^ rightTensor.data)
  }

  override def grad[M <: Nat : ToInt](variable: Variable[V, M]) = {
    val baseGrad: Expression[V, K, Plus[M, CK]] = base.grad(variable)
    val powerGrad: Expression[V, K, Plus[M, CK]] = power.grad(variable)
    val one = ConstantExpression(Tensor.ones(shape.dom, shape.mod))
    baseGrad * (base ^ (power - one)).broadcastCov[M](variable.dom) +
      powerGrad * (Function.log(base) * (base ^ power)).broadcastCov[M](variable.dom)
  }
}

