package emmy.inference

import emmy.autodiff.{Expression, Floating, Node, ValueOps, Variable}
import emmy.distribution.Observation

import scala.annotation.tailrec
import scalaz.Scalaz.Id


case class AEVBModel[V] private[AEVBModel](
                                            nodes: Set[Node],
                                            globalVars: Map[Node, Any]
                                          )(implicit fl: Floating[V], idT: ValueOps[Id, V, Any])
  extends Model[V] {

  type Sampler = AEVBSampler[({type U[_]})#U, V, _]

  import AEVBModel._

  override def update[U[_], S](observations: Seq[Observation[U, V, S]]) = {

    // find new nodes, new variables & their (log) probability
    val (_, samplerBuilders, logp) = collectVars(
      nodes,
      Set.empty[AEVBSamplerBuilder[W forSome {type W[_]}, V, _]],
      fl.zero,
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
      val rho = fl.div(fl.one, fl.fromInt(iter + 10))
      val modelSample = new ModelSample[V] {
        override def getSampleValue[U[_], S](n: Variable[U, V, S]): U[V] =
          variables(n).asInstanceOf[AEVBSampler[U, V, S]].sample()
      }
      val gc = new ModelGradientContext[V](modelSample)
      val updatedWithDelta = samplers.map { sampler =>
        sampler.update(totalLogP, gc, rho)
      }.toMap[Sampler, V]

      val totalDelta = updatedWithDelta.values.sum(fl)
      if (fl.lt(totalDelta, fl.div(fl.one, fl.fromInt(1000)))) {
        updatedWithDelta.keys
      } else {
        iterate(iter + 1, updatedWithDelta.keys)
      }
    }

    val newSamplers = iterate(0, samplers)

    AEVBModel(nodes,
      newSamplers.filter { sampler =>
        globalVars.contains(sampler.variable)
      }.map { sampler =>
        (sampler.variable: Node) -> sampler
      }.toMap
    )(fl, idT)
  }

  override def sample() = new ModelSample[V] {
    override def getSampleValue[U[_], S](n: Variable[U, V, S]): U[V] =
      globalVars(n).asInstanceOf[AEVBSampler[U, V, S]].sample()
  }

  def distributionOf[U[_], V, S](variable: Variable[U, V, S]): (U[V], U[V]) = {
    val sampler = globalVars(variable).asInstanceOf[AEVBSampler[U, V, S]]
    (sampler.mu, sampler.sigma)
  }

}

object AEVBModel {

  def apply[V](global: Seq[Node])(implicit fl: Floating[V]): AEVBModel[V] = {

    // find new nodes, new variables & their (log) probability
    val (_, builders, logp) = collectVars(
      Set.empty,
      Set.empty[AEVBSamplerBuilder[W forSome {type W[_]}, V, _]],
      fl.zero,
      global
    )

    val globalSamplers = initialize(builders, () => {
      new ModelSample[V] {
        override def getSampleValue[U[_], S](n: Variable[U, V, S]): U[V] =
          throw new UnsupportedOperationException("Global priors cannot be initialized with dependencies on variables")
      }
    })

    AEVBModel[V](global.toSet, globalSamplers)
  }

  private[AEVBModel] def initialize[V]
  (
    builders: Set[AEVBSamplerBuilder[W forSome {type W[_]}, V, _]],
    prior: () => ModelSample[V]
  )(implicit idT: ValueOps[Id, V, Any]): Map[Node, Any] = {
    for {_ <- 0 until 100} {
      val modelSample = prior()
      val newVariables = builders.map {
        _.variable: Node
      }
      val ec = new ModelEvaluationContext[V](modelSample, newVariables)
      for {initializer <- builders} {
        initializer.eval(ec)
      }
    }
    builders.toSeq.map { b =>
      b.variable -> b.build()
    }.toMap
  }

  private[AEVBModel] def collectVars[V]
  (
    visited: Set[Node],
    vars: Set[AEVBSamplerBuilder[W forSome {type W[_]}, V, _]],
    lp: Expression[Id, V, Any],
    nodes: Seq[Node]
  ): (Set[Node], Set[AEVBSamplerBuilder[W forSome {type W[_]}, V, _]], Expression[Id, V, Any]) = {
    nodes.foldLeft((visited, vars, lp)) {
      case ((curvis, curvars, curlogp), p) =>
        p match {
          case _ if curvis.contains(p) =>
            (curvis, curvars, curlogp)
          case o: Observation[W forSome {type W[_]}, V, _] =>
            collectVars(curvis + p, curvars, curlogp + o.logp(), p.parents)
          case v: Variable[W forSome {type W[_]}, V, _] =>
            collectVars(curvis + p, curvars + AEVBSamplerBuilder(v), curlogp + v.logp(), p.parents)
          case _ =>
            collectVars(curvis + p, curvars, curlogp, p.parents)
        }
    }
  }
}


