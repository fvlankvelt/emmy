package emmy.inference

import emmy.autodiff.ContainerOps.Aux
import emmy.autodiff.{ CategoricalVariable, ContinuousVariable, Expression, GradientContext, Node, Variable }

import scala.collection.mutable
import scalaz.Scalaz.Id

class ModelGradientContext(model: Model, deps: Map[Node, Set[Node]] = Map.empty) extends GradientContext {

  private val modelSample = model.sample(this)
  private val cache = mutable.HashMap[AnyRef, Any]()

  override def apply[U[_], V, S](n: Expression[U, V, S]): U[V] =
    n match {
      case v: Variable[U, V, S] if v.isInstanceOf[CategoricalVariable] ⇒
        val value = modelSample.getSampleValue[Id, Int, Any](v.asInstanceOf[CategoricalVariable])
        cache.getOrElseUpdate(n, value)
          .asInstanceOf[U[V]]
      case v: ContinuousVariable[U, S] ⇒
        cache.getOrElseUpdate(n, modelSample.getSampleValue[U, Double, S](v))
          .asInstanceOf[U[V]]
      case _ ⇒
        cache.getOrElseUpdate(n, n.apply(this))
          .asInstanceOf[U[V]]
    }

  override def apply[W[_], U[_], V, T, S](
    n: Expression[U, V, S],
    v: ContinuousVariable[W, T]
  )(implicit wOps: Aux[W, T]): Option[W[U[Double]]] = {
    val eval = deps.get(v).forall {
      _.contains(n)
    }
    if (eval) {
      n.grad(this, v)
    }
    else {
      None
    }
    //    val result = n.grad(this, v)
    //    if (!eval && result.isDefined) {
    //      throw new Exception("Expression is not evaluated, but should be")
    //    }
    //    result
  }
}
