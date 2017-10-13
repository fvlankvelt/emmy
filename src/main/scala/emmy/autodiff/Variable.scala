package emmy.autodiff

import emmy.distribution.Stochast


trait Variable[U[_], S] extends Expression[U, Double, S] with Stochast {

//  override def apply(evaluationContext: EvaluationContext): U[V] =
//    throw new UnsupportedOperationException("Evaluation context should provide value for variable")

  override def grad[W[_], T](gc: GradientContext, v: Variable[W, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val shape = wOps.shapeOf(gc(v))
    val valT = vt(gc)
    if (this == v) {
      wOps.eye(shape, valT.valueVT.one, valT.valueVT.zero).asInstanceOf[Gradient[W, U]]
    } else {
      wOps.fill(shape, valT.zero)
    }
  }
}

