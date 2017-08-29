package pp.tensor

case class Normal[
K <: Nat,
CK <: Nat
](mu: Expression[K, CK], sigma: Expression[K, CK])
  extends Expression[K, CK] {

  assert(mu.shape == sigma.shape)

  override val shape = mu.shape
  private val ones = ConstantExpression(Tensor.ones(shape.dom, shape.mod))

  private val tau = ones / (sigma * sigma)

  def logp() = {
    val delta = this - mu
    val exponent = -tau * delta * delta
    val norm = Function.log(tau / (2 * Math.PI.toFloat))
    Function.sum(exponent + norm) / 2f
  }

  override def eval() = throw new NotImplementedError("A distribution cannot be evaluated - it's values must be provided by the engine")

  override def grad[M <: Nat : ToInt](variable: Variable[M]) = ???
}
