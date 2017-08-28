package pp.tensor

import breeze.math.Field
import pp.tensor.Function.ValueConverter

import scala.reflect.ClassTag

case class Normal[
V: ClassTag : Field : ValueConverter,
K <: Nat,
CK <: Nat
](mu: Expression[V, K, CK], sigma: Expression[V, K, CK])
  extends Expression[V, K, CK] {

  assert(mu.shape == sigma.shape)

  val ringV = implicitly[Field[V]]
  val ctV = implicitly[ClassTag[V]]

  override val shape = mu.shape
  private val ones = ConstantExpression(Tensor.ones(shape.dom, shape.mod))

  private val tau = ones / (sigma * sigma)

  def logp() = {
    val delta = this - mu
    val exponent = -tau * delta * delta
    val norm = Function.log(tau / (2 * Math.PI))
    Function.sum(exponent + norm) / 2f
  }

  override def eval() = throw new NotImplementedError("A distribution cannot be evaluated - it's values must be provided by the engine")

  override def grad[M <: Nat : ToInt](variable: Variable[V, M]) = ???
}
