package emmy.inference.aevb

import emmy.autodiff._
import emmy.distribution.{ Categorical, Factor, Normal, Observation }
import emmy.inference._
import emmy.inference.aevb.AEVBModel.{ ParameterOptimizer, VariablePosterior }
import shapeless.Id

import scala.annotation.tailrec

/*
class AEVBModel private[AEVBModel](
                                    nodes: Set[Node],
                                    globalVars: Map[Node, SamplerBuilder],
                                    globalDeps: Map[Node, Set[Node]]
                                  )
  extends Model {

  import AEVBModel._

  override def sample[U[_], V, S](n: Variable[U, V, S], ec: SampleContext): U[V] = {
    n match {
      case _: ContinuousVariable[U, S] ⇒
        globalVars(n).asInstanceOf[ContinuousSampler[U, S]].sample(ec).asInstanceOf[U[V]]
      case v if v.isInstanceOf[CategoricalVariable] ⇒
        globalVars(n).asInstanceOf[CategoricalSampler].sample(ec).asInstanceOf[U[V]]
    }
  }

  private[aevb] def getSampler[U[_], V, S](node: Node) =
    globalVars(node).asInstanceOf[ContinuousSampler[U, S]]

  override def update[U[_], V, S](observations: Seq[Observation[U, V, S]]) = {

    // what dependencies are needed here?
    val globalCtx = new ModelGradientContext(this, globalDeps)

    // find new nodes, new variables & their (log) probability
    val (_, samplerBuilders, factors, deps) = {
      val collector = new VariableCollector(nodes, globalDeps)
      collector.collect(observations ++ globalVars.keys)
    }

    // initialize new variables by sampling their prior
    // (based on current distributions for already known variables)
    val localVars = if (samplerBuilders.nonEmpty) {
      initialize(samplerBuilders)
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

    val samplers = (globalVars ++ localVars).values.map {
      _.build(logP.values.toSeq)
    }
    refine(0, samplers)

    new AEVBModel(
      nodes,
      samplers
        .filter { sampler ⇒ globalVars.contains(sampler.variable) }
        .map { sampler ⇒ (sampler.variable: Node) -> sampler }
        .toMap
        .mapValues {
          _.toBuilder(globalCtx)
        },
      globalDeps
    )
  }

  @tailrec
  private def refine(iter: Int, samplers: Iterable[Sampler]): Unit = {
    val rho = 1.0 / (iter + 1)
    val sc = SampleContext(iter, iter)
    val deltas = samplers.map { sampler ⇒
      sampler.update(sc, rho)
    }

    val totalDelta = deltas.sum
    if (totalDelta > 0.0001) {
      refine(iter + 1, samplers)
    }
  }

  def distributionOf[U[_], S](variable: ContinuousVariable[U, S]): (U[Double], U[Double]) = {
    val sampler = globalVars(variable).asInstanceOf[ContinuousSampler[U, S]]
    (sampler.mu, sampler.sigma)
  }

}
*/

object GlobalModel extends Model {
  override def sample[U[_], V, S](v: Variable[U, V, S], ec: SampleContext) =
    throw new UnsupportedOperationException("Global priors cannot be initialized with dependencies on variables")
}

case class AEVBModel(variables: Seq[VariablePosterior], parameters: Seq[ParameterOptimizer]) extends Model {

  override def sample[U[_], V, S](n: Variable[U, V, S], ec: SampleContext): U[V] = ???

  override def update[U[_], V, S](observations: Seq[Observation[U, V, S]]) = {
    this
  }

  def distributionOf[U[_], S](variable: ContinuousVariable[U, S]): (U[Double], U[Double]) = ???
}

object AEVBModel {

  trait ParameterOptimizer {

    def initialize(target: Expression[Id, Double, Any], gc: GradientContext): Unit

    def update(ctx: SampleContext): Unit
  }

  //@formatter:off
  /**
   * Simple stochastic gradient descent
   * Takes the gradient of log(P) - log(Q) to a parameter.
   *
   * For continuous variables, parameters describing a distribution can be more effectively optimized by using the
   * "path derivative" from
   *  Sticking the Landing: Simple, Lower-Variance Gradient Estimators for Variational Inference
   *    Geoffrey Roeder, Yuhuai Wu, David Duvenaud - https://arxiv.org/abs/1703.09194
   * I.e. take as gradient the derivative with respect to the random variable, multiplied by the derivative of the
   * random variable with respect to the parameter.  This eliminates the (high-variance) score function.
   */
  //@formatter:on
  case class ParameterHolder[U[_], S](parameter: Parameter[U, S])
    extends ParameterOptimizer {

    private var valueEv: Evaluable[U[Double]] = null
    private var gradient: Gradient[U, Id] = None
    var value: Option[U[Double]] = None

    def initialize(target: Expression[Id, Double, Any], gc: GradientContext) = {
      valueEv = parameter.eval(gc)
      gradient = target.grad(gc, parameter)
    }

    def update(ctx: SampleContext) = {
      if (value.isEmpty) {
        value = Some(valueEv(ctx))
      }

      gradient.map { gradEval ⇒
        gradEval(ctx)
      }.foreach { gradValue ⇒
        val rho = 1.0 / (ctx.iteration + 1)
        val vt = parameter.vt(ctx)
        val ops = parameter.ops
        value = value.map { v ⇒
          vt.plus(v, ops.map(gradValue) {
            _ * rho
          })
        }
        //        println(s"Setting ${parameter.hashCode} to ${value.get}")
        parameter.v = value.get
      }
    }
  }

  sealed trait VariablePosterior {
    def P: Factor
    def Q: Factor
  }

  case class ContinuousVariablePosterior[U[_], S](variable: Variable[U, Double, S])
    extends VariablePosterior {

    override val P: Variable[U, Double, S] = variable

    private implicit val ops: ContainerOps.Aux[U, S] = variable.ops
    val mu = new Parameter(variable.vt.map { vo ⇒ vo.zero })(Floating.doubleFloating, variable.so, variable.ops)
    val sigma = exp(new Parameter(variable.vt.map { vo ⇒ vo.zero })(Floating.doubleFloating, variable.so, variable.ops))

    override val Q: Variable[U, Double, S] = Normal(mu, sigma).sample
  }

  case class CategoricalVariablePosterior(variable: CategoricalVariable)
    extends VariablePosterior {

    override val P: CategoricalVariable = variable

    private implicit val ops: ContainerOps.Aux[Id, Any] = variable.ops
    val thetas = new Parameter[IndexedSeq, Int](variable.K.map { k ⇒
      Array.fill(k)(1.0 / k): IndexedSeq[Double]
    })

    override val Q: CategoricalVariable = Categorical(thetas).sample
  }

  /**
   * Create a new model with a global set of variables to infer
   */
  def apply(global: Seq[Node]): AEVBModel = {

    // find new nodes, new variables & their (log) probability
    val collector = new VariableCollector(Set.empty, Map.empty)
    val (_, variables, parameters, factors, globalDeps) = collector.collect(global)

    // no optimizing hyper-parameters
    //    assert(parameters.isEmpty)

    val logp = variables.map {
      _.P.logp
    }.reduceOption(_ + _)
      .getOrElse(Constant(0.0))

    val logq = variables.map {
      _.Q.logp
    }.reduceOption(_ + _)
      .getOrElse(Constant(0.0))

    val gc = new ModelGradientContext(
      variables.map { v ⇒
        (v.P: Node, v)
      }.toMap,
      Map.empty
    )
    parameters.foreach(param ⇒
      param.initialize(logp - logq, gc))
    for {
      iter ← Range(0, 10000)
      ctx = SampleContext(iter, iter)
      param ← parameters
    } {
      param.update(ctx)
    }

    //    val globalSamplers = initialize(builders)
    AEVBModel(variables.toSeq, parameters.toSeq)
  }

  @tailrec
  private def refine(iter: Int, samplers: Iterable[Sampler]): Unit = {
    val rho = 1.0 / (iter + 1)
    val sc = SampleContext(iter, iter)
    val deltas = samplers.map { sampler ⇒
      sampler.update(sc, rho)
    }

    val totalDelta = deltas.sum
    if (totalDelta > 0.0001) {
      refine(iter + 1, samplers)
    }
  }

  /*
  private[AEVBModel] def initialize(
                                     builders: Set[SamplerInitializer]
                                   ): Map[Node, SamplerInitializer] = {
    for {iter ← 0 until 100} {
      val ctx = SampleContext(iter, iter)
      for {builder ← builders} {
        builder.eval(ctx)
      }
    }
    builders.toSeq.map { b ⇒
      b.variable -> b
    }.toMap
  }
  */

  class VariableCollector(
      var visited: Set[Node],
      var deps:    Map[Node, Set[Node]]
  )
    extends Visitor[VariableCollector] {

    var parameters: Set[ParameterOptimizer] = Set.empty
    var variables: Set[VariablePosterior] = Set.empty
    var factors: Seq[Factor] = Seq.empty

    // stack of nodes, from the end result down
    var stack: List[Node] = List.empty

    private[aevb] def collectVars(nodes: Seq[Node]): VariableCollector = {
      nodes.foreach { p ⇒
        if (visited.contains(p)) {
          //          println(": FOUND")
          val newdeps = deps.map {
            case (variable, dependants) ⇒
              if (dependants.contains(p)) {
                variable -> (dependants ++ stack.toSet)
              }
              else {
                variable -> dependants
              }
          }
          deps = newdeps
        }
        else {
          visited += p
          stack = p :: stack
          p.visit(this)
          stack = stack.tail
        }
      }
      this
    }

    def collect(nodes: Seq[Node]): (Set[Node], Set[VariablePosterior], Set[ParameterOptimizer], Seq[Factor], Map[Node, Set[Node]]) = {
      collectVars(nodes) // ignore result
      (visited, variables, parameters, factors, deps)
    }

    override def visitParameter[U[_], S](p: Parameter[U, S]): VariableCollector = {
      parameters += ParameterHolder(p)
      this
    }

    override def visitSampler(o: Sampler): VariableCollector = {
      factors :+= o
      collectVars(o.parents)
    }

    override def visitObservation[U[_], V, S](o: Observation[U, V, S]): VariableCollector = {
      factors :+= o
      collectVars(o.parents)
    }

    override def visitContinuousVariable[U[_], S](v: ContinuousVariable[U, S]): VariableCollector = {
      factors :+= v
      val posterior = ContinuousVariablePosterior(v)
      variables += posterior
      deps += v -> stack.toSet
      collectVars(v.parents ++ posterior.Q.parents)
    }

    override def visitCategoricalVariable(v: CategoricalVariable): VariableCollector = {
      factors :+= v
      val posterior = CategoricalVariablePosterior(v)
      variables += posterior
      deps += v -> stack.toSet
      collectVars(v.parents ++ posterior.Q.parents)
    }

    override def visitNode(n: Node): VariableCollector =
      collectVars(n.parents)

  }

}

