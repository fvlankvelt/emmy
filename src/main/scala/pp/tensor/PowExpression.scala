package pp.tensor

case class PowExpression[
K <: Nat,
CK <: Nat
](base: Expression[K, CK], power: Expression[K, CK])
  extends Expression[K, CK] {

  assert(base.shape == power.shape)

  override val shape = base.shape

  override def eval() = {
    val leftTensor = base.eval()
    val rightTensor = power.eval()
    Tensor[K, CK](shape.dom, shape.mod, leftTensor.data ^:^ rightTensor.data)
  }

  override def grad[M <: Nat : ToInt](variable: Variable[M]) = {
    val baseGrad: Expression[K, Plus[M, CK]] = base.grad(variable)
    val powerGrad: Expression[K, Plus[M, CK]] = power.grad(variable)
    val one = ConstantExpression(Tensor.ones(shape.dom, shape.mod))
    baseGrad * (base ^ (power - one)).broadcastCov[M](variable.dom) +
      powerGrad * (Function.log(base) * (base ^ power)).broadcastCov[M](variable.dom)
  }
}

