package emmy.inference

import emmy.autodiff.{EvaluationContext, Expression, Node, Variable}
import emmy.distribution.Observation

import scala.annotation.tailrec
import scalaz.Scalaz.Id


class AEVBSamplersModel(globalVars: Map[Node, Any]) extends Model {

  override def sample(ec: EvaluationContext) = new ModelSample {
    override def getSampleValue[U[_], S](n: Variable[U, S]) =
      globalVars(n).asInstanceOf[AEVBSampler[U, S]].sample(ec)
  }

}

class AEVBModel private[AEVBModel](
                                       nodes: Set[Node],
                                       globalVars: Map[Node, Any]
                                     )
  extends AEVBSamplersModel(globalVars) {

  type Sampler = AEVBSampler[({type U[_]})#U, _]

  import AEVBModel._

  def getSampler[U[_], V, S](node: Node) = globalVars(node).asInstanceOf[AEVBSampler[U, S]]

  override def update[U[_], V, S](observations: Seq[Observation[U, V, S]]) = {

    // find new nodes, new variables & their (log) probability
    val (_, samplerBuilders, logp) = collectVars(
      nodes,
      Set.empty[AEVBSamplerBuilder[W forSome {type W[_]}, _]],
      0.0,
      observations
    )

    // initialize new variables by sampling their prior
    // (based on current distributions for already known variables)
    val localVars = if (samplerBuilders.nonEmpty) {
      initialize(samplerBuilders, sample)
    } else {
      Map.empty
    }

    // logP is the sum of
    // - the likelihood of observation log(p(x|\theta)), and
    // - the prior log(p(\theta)) of the local variables
    //
    // Add the log prior of global variables to get the full
    // objective function to optimize
    val totalLogP = globalVars.values.foldLeft(logp) { case (curLogp, variable) =>
      curLogp + variable.asInstanceOf[Sampler].logp()
    }

    // update variables by taking observations into account
    val samplers = (globalVars ++ localVars).map {
      _._2.asInstanceOf[Sampler]
    }

    @tailrec
    def iterate(iter: Int, samplers: Iterable[Sampler]): Iterable[Sampler] = {
      val variables = samplers.map { s => (s.variable: Node) -> (s: Any) }.toMap
      val rho = 1.0 / (iter + 10)
      val model = new AEVBSamplersModel(variables)
      val gc = new ModelGradientContext(model)
      val updatedWithDelta = samplers.map { sampler =>
        sampler.update(totalLogP, gc, rho)
      }.toMap[Sampler, Double]

      val totalDelta = updatedWithDelta.values.sum
      if (totalDelta < 0.001) {
        updatedWithDelta.keys
      } else {
        iterate(iter + 1, updatedWithDelta.keys)
      }
    }

    val newSamplers = iterate(0, samplers)

    new AEVBModel(nodes,
      newSamplers.filter { sampler =>
        globalVars.contains(sampler.variable)
      }.map { sampler =>
        (sampler.variable: Node) -> sampler
      }.toMap
    )
  }

  def distributionOf[U[_], S](variable: Variable[U, S]): (U[Double], U[Double]) = {
    val sampler = globalVars(variable).asInstanceOf[AEVBSampler[U, S]]
    (sampler.mu, sampler.sigma)
  }

}

object AEVBModel {

  def apply(global: Seq[Node]): AEVBModel = {

    // find new nodes, new variables & their (log) probability
    val (_, builders, logp) = collectVars(
      Set.empty,
      Set.empty[AEVBSamplerBuilder[W forSome {type W[_]}, _]],
      0.0,
      global
    )

    val globalSamplers = initialize(builders, (ec: EvaluationContext) => {
      new ModelSample {
        override def getSampleValue[U[_], S](n: Variable[U, S]): U[Double] =
          throw new UnsupportedOperationException("Global priors cannot be initialized with dependencies on variables")
      }
    })

    new AEVBModel(global.toSet, globalSamplers)
  }

  private[AEVBModel] def initialize
  (
    builders: Set[AEVBSamplerBuilder[W forSome {type W[_]}, _]],
    prior: EvaluationContext => ModelSample
  ): Map[Node, Any] = {
    for {_ <- 0 until 100} {
      val variables = builders.map {
        _.variable: Node
      }
      val ec = new ModelEvaluationContext {
        val newVariables = variables
        val modelSample = prior(this: EvaluationContext)
      }
      for {initializer <- builders} {
        initializer.eval(ec)
      }
    }
    builders.toSeq.map { b =>
      b.variable -> b.build()
    }.toMap
  }

  private[AEVBModel] def collectVars
  (
    visited: Set[Node],
    vars: Set[AEVBSamplerBuilder[W forSome {type W[_]}, _]],
    lp: Expression[Id, Double, Any],
    nodes: Seq[Node]
  ): (Set[Node], Set[AEVBSamplerBuilder[W forSome {type W[_]}, _]], Expression[Id, Double, Any]) = {
    nodes.foldLeft((visited, vars, lp)) {
      case ((curvis, curvars, curlogp), p) =>
        p match {
          case _ if curvis.contains(p) =>
            (curvis, curvars, curlogp)
          case o: Observation[W forSome {type W[_]}, _, _] =>
            collectVars(curvis + p, curvars, curlogp + o.logp(), p.parents)
          case v: Variable[W forSome {type W[_]}, _] =>
            collectVars(curvis + p, curvars + AEVBSamplerBuilder(v), curlogp + v.logp(), p.parents)
          case _ =>
            collectVars(curvis + p, curvars, curlogp, p.parents)
        }
    }
  }
}


