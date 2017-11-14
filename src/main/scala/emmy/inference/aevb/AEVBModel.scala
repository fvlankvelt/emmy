package emmy.inference.aevb

import emmy.autodiff._
import emmy.distribution.{ Factor, Observation }
import emmy.inference._

import scala.annotation.tailrec
import scalaz.Scalaz.Id

class AEVBSamplersModel(globalVars: Map[Node, Sampler]) extends Model {

  override def sample(ec: EvaluationContext) = new ModelSample {
    override def getSampleValue[U[_], V, S](n: Variable[U, V, S]) =
      n match {
        case _: ContinuousVariable[U, S] ⇒
          globalVars(n).asInstanceOf[ContinuousSampler[U, S]].sample(ec).asInstanceOf[U[V]]
        case v if v.isInstanceOf[CategoricalVariable] ⇒
          globalVars(n).asInstanceOf[CategoricalSampler].sample(ec).asInstanceOf[U[V]]
      }
  }

}

class AEVBModel private[AEVBModel] (
    nodes:      Set[Node],
    globalVars: Map[Node, Sampler],
    globalDeps: Map[Node, Set[Node]]
)
  extends AEVBSamplersModel(globalVars) {

  import AEVBModel._

  def getSampler[U[_], V, S](node: Node) = globalVars(node).asInstanceOf[ContinuousSampler[U, S]]

  override def update[U[_], V, S](observations: Seq[Observation[U, V, S]]) = {

    // find new nodes, new variables & their (log) probability
    val (knownNodes, samplerBuilders, factors, deps) = collectVars(
      nodes,
      Set.empty[SamplerBuilder],
      Seq.empty,
      observations ++ globalVars.values,
      Seq.empty,
      globalDeps
    )

    // initialize new variables by sampling their prior
    // (based on current distributions for already known variables)
    val localVars = if (samplerBuilders.nonEmpty) {
      initialize(samplerBuilders, sample)
    }
    else {
      Map.empty
    }

    // logP is the sum of
    // - the likelihood of observation log(p(x|\theta)), and
    // - the prior log(p(\theta)) of the local variables
    //
    // Add the log prior of global variables to get the full
    // objective function to optimize
    val logP = factors.map { factor ⇒
      (factor: Node) -> factor.logp
    }.toMap

    // traverse the logp graphs to collect additional nodes
    val (_, _, _, fullDeps) = collectVars(
      knownNodes,
      Set.empty,
      Seq.empty,
      logP.values.toSeq,
      Seq.empty,
      deps
    )

    // update variables by taking observations into account
    val samplers = (globalVars ++ localVars).map {
      case (variable, sampler) ⇒
        val dependencies = deps.getOrElse(variable, Set.empty)
        val filtered = logP.flatMap {
          case (factor, logp) ⇒
            if (dependencies.contains(factor)) Some(logp) else None
        }
        (
          sampler.asInstanceOf[Sampler],
          filtered.toSeq
        )
    }

    @tailrec
    def iterate(iter: Int, samplers: Iterable[(Sampler, Seq[Expression[Id, Double, Any]])]): Iterable[Sampler] = {
      val variables = samplers.map { s ⇒ s._1.variable -> s._1 }.toMap
      val rho = 1.0 / (iter + 1)
      val model = new AEVBSamplersModel(variables)
      val gc = new ModelGradientContext(model, fullDeps)
      val updatedWithDelta = samplers.map { sampler ⇒
        val (newSampler, delta) = sampler._1.update(sampler._2, gc, rho)
        ((newSampler, sampler._2), delta)
      }

      val totalDelta = updatedWithDelta.map(_._2).sum
      if (totalDelta < 0.0001) {
        updatedWithDelta.map(_._1._1)
      }
      else {
        iterate(iter + 1, updatedWithDelta.map(_._1))
      }
    }

    val newSamplers = iterate(0, samplers)

    new AEVBModel(
      nodes,
      newSamplers.filter { sampler ⇒
        globalVars.contains(sampler.variable)
      }.map { sampler ⇒
        (sampler.variable: Node) -> sampler
      }.toMap,
      globalDeps
    )
  }

  def distributionOf[U[_], S](variable: ContinuousVariable[U, S]): (U[Double], U[Double]) = {
    val sampler = globalVars(variable).asInstanceOf[ContinuousSampler[U, S]]
    (sampler.mu, sampler.sigma)
  }

}

object AEVBModel {

  def apply(global: Seq[Node]): AEVBModel = {

    // find new nodes, new variables & their (log) probability
    val (_, builders, _, globalDeps) = collectVars(
      Set.empty,
      Set.empty[SamplerBuilder],
      Seq.empty,
      global
    )

    val globalSamplers = initialize(builders, (ec: EvaluationContext) ⇒ {
      new ModelSample {
        override def getSampleValue[U[_], V, S](n: Variable[U, V, S]): U[V] =
          throw new UnsupportedOperationException("Global priors cannot be initialized with dependencies on variables")
      }
    })

    new AEVBModel(global.toSet, globalSamplers, globalDeps)
  }

  private[AEVBModel] def initialize(
    builders: Set[SamplerBuilder],
    prior:    EvaluationContext ⇒ ModelSample
  ): Map[Node, Sampler] = {
    for { _ ← 0 until 100 } {
      val variables = builders.map {
        _.variable: Node
      }
      val ec = new ModelEvaluationContext {
        val newVariables = variables
        val modelSample = prior(this: EvaluationContext)
      }
      for { initializer ← builders } {
        initializer.eval(ec)
      }
    }
    builders.toSeq.map { b ⇒
      b.variable -> b.build()
    }.toMap
  }

  private[aevb] def collectVars(
    visited: Set[Node],
    vars:    Set[SamplerBuilder],
    factors: Seq[Factor],
    nodes:   Seq[Node],
    // stack of nodes, from the end result down
    stack: Seq[Node] = Seq.empty,
    // set of expressions that depend on a variable, by variable
    deps: Map[Node, Set[Node]] = Map.empty
  ): (Set[Node], Set[SamplerBuilder], Seq[Factor], Map[Node, Set[Node]]) = {
    nodes.foldLeft((visited, vars, factors, deps)) {
      case ((curvis, curvars, curFactors, curdeps), p) ⇒
        //        print(stack.map(_ => "    ").mkString("") + p.toString)
        if (curvis.contains(p)) {
          //          println(": FOUND")
          val newdeps = curdeps.map {
            case (variable, dependants) ⇒
              if (dependants.contains(p)) {
                variable -> (dependants ++ stack.toSet)
              }
              else {
                variable -> dependants
              }
          }
          (
            curvis,
            curvars,
            curFactors,
            newdeps
          )
        }
        else {
          //          println(": NEW")
          p.visit(new Visitor[(Set[Node], Set[SamplerBuilder], Seq[Factor], Map[Node, Set[Node]])] {

            override def visitObservation[U[_], V, S](o: Observation[U, V, S]) = {
              collectVars(
                curvis + p,
                curvars,
                curFactors :+ o,
                p.parents,
                stack :+ p,
                curdeps
              )
            }

            override def visitSampler(o: Sampler) = {
              collectVars(
                curvis + p,
                curvars,
                curFactors :+ o,
                p.parents,
                stack :+ p,
                curdeps
              )
            }

            override def visitContinuousVariable[U[_], S](v: ContinuousVariable[U, S]) = {
              collectVars(
                curvis + p,
                curvars + ContinuousSamplerBuilder(v),
                curFactors :+ v,
                p.parents,
                stack :+ p,
                curdeps + (p -> (stack.toSet + v))
              )
            }

            override def visitCategoricalVariable(v: CategoricalVariable) = {
              collectVars(
                curvis + p,
                curvars + CategoricalSamplerBuilder(v),
                curFactors :+ v,
                p.parents,
                stack :+ p,
                curdeps + (p -> (stack.toSet + v))
              )
            }

            override def visitNode(n: Node) =
              collectVars(
                curvis + n,
                curvars,
                curFactors,
                n.parents,
                stack :+ p,
                curdeps
              )

          })
        }
    }
  }
}

