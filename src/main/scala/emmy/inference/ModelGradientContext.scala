package emmy.inference

import emmy.autodiff.ContainerOps.Aux
import emmy.autodiff.{Expression, GradientContext, Variable}

import scala.collection.mutable

class ModelGradientContext(model: Model) extends GradientContext {

  private val modelSample = model.sample(this)
  private val cache = mutable.HashMap[AnyRef, Any]()

  override def apply[U[_], V, S](n: Expression[U, V, S]): U[V] =
    n match {
      case v: Variable[U, S] =>
        cache.getOrElseUpdate(n, modelSample.getSampleValue[U, S](v))
          .asInstanceOf[U[V]]
      case _ =>
        cache.getOrElseUpdate(n, n.apply(this))
          .asInstanceOf[U[V]]
    }

  override def apply[W[_], U[_], V, T, S](n: Expression[U, V, S], v: Variable[W, T])(implicit wOps: Aux[W, T]): W[U[Double]] = {
    n.grad(this, v)
  }
}
