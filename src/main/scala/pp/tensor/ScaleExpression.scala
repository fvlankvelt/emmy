package pp.tensor

case class ScaleExpression[
K <: Nat,
CK <: Nat
](upstream: Expression[K, CK], scale: Float)
  extends Expression[K, CK] {

  override val shape = upstream.shape

  override def eval() = {
    Tensor[K, CK](shape.dom, shape.mod, scale * upstream.eval().data)
  }

  override def grad[M <: Nat : ToInt](variable: Variable[M]) = {
    scale * upstream.grad(variable)
  }
}

