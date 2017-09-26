package emmy.inference

import emmy.autodiff.ContainerOps.Aux
import emmy.autodiff.{Expression, GradientContext, Variable}

import scala.collection.mutable

class ModelGradientContext(modelSample: ModelSample) extends GradientContext {

  private val cache = mutable.HashMap[AnyRef, Any]()

  override def apply[U[_], V, S](n: Expression[U, V, S]): U[V] =
    n match {
      case v: Variable[U, V, S] =>
        cache.getOrElseUpdate(n, modelSample.getSampleValue(v))
          .asInstanceOf[U[V]]
      case _ =>
        cache.getOrElseUpdate(n, n.apply(this))
          .asInstanceOf[U[V]]
    }

  override def apply[W[_], U[_], V, T, S](n: Expression[U, V, S], v: Variable[W, V, T])(implicit wOps: Aux[W, T]): W[U[V]] = {
    n.grad(this, v)
  }
}
