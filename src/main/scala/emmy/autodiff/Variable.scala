package emmy.autodiff

import emmy.distribution.Stochast


trait Variable[U[_], V, S] extends Node[U, V, S] with Stochast[V] {

  override def apply(evaluationContext: EvaluationContext): U[V] =
    throw new UnsupportedOperationException("Evaluation context should provide value for variable")

  override def grad[W[_], T](gc: GradientContext, v: Variable[W, V, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val ops = implicitly[ContainerOps[W]]
    val shape = ops.shapeOf(gc(v))
    if (this == v) {
      ops.eye(shape, vt.valueVT.one, vt.valueVT.zero).asInstanceOf[Gradient[W, U, V]]
    } else {
      ops.fill(shape, vt.zero)
    }
  }
}

