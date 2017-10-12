package emmy.inference

import emmy.autodiff.{EvaluationContext, Expression, Node, Variable}

import scala.collection.mutable

trait ModelEvaluationContext
  extends EvaluationContext {

  def newVariables: Set[Node]
  def modelSample: ModelSample

  private val cache = mutable.HashMap[AnyRef, Any]()

  override def apply[U[_], V, S](n: Expression[U, V, S]): U[V] =
    n match {
      case v: Variable[U, S] if !newVariables.contains(v) =>
        cache.getOrElseUpdate(n, modelSample.getSampleValue[U, S](v))
          .asInstanceOf[U[V]]
      case _ =>
        cache.getOrElseUpdate(n, n.apply(this))
          .asInstanceOf[U[V]]
    }

}
