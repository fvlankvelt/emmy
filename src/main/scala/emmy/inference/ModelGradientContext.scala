package emmy.inference

import breeze.numerics.abs
import emmy.autodiff.{Evaluable, Expression, Gradient, GradientContext, Node, Parameter, SampleContext, Variable}
import emmy.inference.aevb.AEVBModel.VariablePosterior

import scala.collection.mutable

class ModelGradientContext(
    posteriors:   Map[Node, VariablePosterior],
    dependencies: Map[Node, Set[Node]]         = Map.empty
)
  extends GradientContext {

  private val cache = mutable.HashMap[AnyRef, Any]()

  override def apply[U[_], V, S](n: Expression[U, V, S]): Evaluable[U[V]] =
    n match {
      case v: Variable[U, V, S] if posteriors.contains(v) ⇒
        val q = posteriors(v).Q.asInstanceOf[Variable[U, V, S]]
        cache.getOrElseUpdate(n, apply(q)).asInstanceOf[Evaluable[U[V]]]
      case _ ⇒
        cache.getOrElseUpdate(n, wrap(n.eval(this), n.toString)).asInstanceOf[Evaluable[U[V]]]
    }

  private def wrap[U[_], V](eval: Evaluable[U[V]], name: String): Evaluable[U[V]] = new Evaluable[U[V]] {

    private var lastContext = -1
    private var lastValue: Option[U[V]] = None

    override def apply(ec: SampleContext): U[V] = {
      if (ec.iteration != lastContext) {
        lastValue = Some(eval(ec))
//        println(s"Setting ${name} to ${lastValue.get}")
        val asDouble = lastValue.get.asInstanceOf[Double]
        val asStr = asDouble.toString
        if (asStr == "NaN" || asStr == "Infinity" || abs(asDouble) > 10.0) {
          if (asStr == "NaN" || asStr == "Infinity")
            assert(false)
        }
        lastContext = ec.iteration
      }
      lastValue.get
    }

  }

  override def apply[W[_], U[_], V, T, S](
    n: Expression[U, V, S],
    v: Parameter[W, T]
  ): Gradient[W, U] = {
    val eval = dependencies.get(v).forall {
      _.contains(n)
    }
    if (eval) {
      if (posteriors.contains(n)) {
        val q = posteriors(n).Q
          .asInstanceOf[Variable[U, V, S]]
        val g = q.grad(this, v)
        println(s"Replacing ${n} by ${q}")
        g
      }
      else
        n.grad(this, v)
    }
    else {
      None
    }
    /*
    val result = n.grad(this, v)
    if (!eval && result.isDefined) {
      throw new Exception("Expression is not evaluated, but should be")
    }
    result
    */
  }
}
