package emmy.autodiff

import emmy.distribution.Stochast


trait Variable[U[_], V, S] extends Expression[U, V, S] with Stochast

trait ContinuousVariable[U[_], S] extends Variable[U, Double, S] {

  override def grad[W[_], T](gc: GradientContext, v: ContinuousVariable[W, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val shape = wOps.shapeOf(gc(v))
    val valT = vt(gc)
    if (this == v) {
      wOps.eye(shape, valT.valueVT.one, valT.valueVT.zero).asInstanceOf[Gradient[W, U]]
    } else {
      wOps.fill(shape, valT.zero)
    }
  }
}

