package emmy.inference

import emmy.autodiff.{EvaluationContext, Expression, Node, Variable}

import scala.collection.mutable

class ModelEvaluationContext[V](modelSample: ModelSample[V], newVariables: Set[Node]) extends EvaluationContext[V] {

  private val cache = mutable.HashMap[AnyRef, Any]()

  override def apply[U[_], S](n: Expression[U, V, S]): U[V] =
    n match {
      case v: Variable[U, V, S] if !newVariables.contains(v) =>
        cache.getOrElseUpdate(n, modelSample.getSampleValue(v))
          .asInstanceOf[U[V]]
      case _ =>
        cache.getOrElseUpdate(n, n.apply(this))
          .asInstanceOf[U[V]]
    }

}
