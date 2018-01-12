package emmy.inference.aevb

import breeze.numerics.abs
import emmy.autodiff._
import emmy.distribution.{Categorical, Factor, Normal, Observation}
import emmy.inference._
import emmy.inference.aevb.AEVBModel.{ParameterHolder, VariableCollector, VariablePosterior}

import scala.annotation.tailrec
import scalaz.Scalaz.Id

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

case class AEVBModel(variables: Set[VariablePosterior]) extends Model {

  override def sample[U[_], V, S](n: Variable[U, V, S], ec: SampleContext): U[V] = ???

  override def update[U[_], V, S](observations: Seq[Observation[U, V, S]]) = {
    // find new nodes, new variables & their (log) probability
    val collector = new VariableCollector(variables.map { v => v.O }.toSet, Map.empty)
    val (_, localVars, localParams, factors, globalDeps) = collector.collect(observations)

    val newGlobal = variables.map { _.next }

    val allVars = newGlobal ++ localVars
    val logp = (factors ++ allVars.map { _.P })
        .map {_.logp}
        .reduceOption(_ + _)
        .getOrElse(Constant(0.0))
    val logq = allVars.map {
      _.Q.logp
    }.reduceOption(_ + _)
      .getOrElse(Constant(0.0))

    val newGlobalParams = newGlobal.flatMap(_.parameters)
    val allParams = newGlobalParams ++ localParams
    val gc = new ModelGradientContext(
      allVars.flatMap { v ⇒
          (v.O: Node, v) :: (v.P: Node, v) :: Nil
        }.toMap,
      Map.empty
    )
    val ctx = SampleContext(0, 0)

    allParams.foreach(_.initialize(logp - logq, gc, ctx))
    var iter = 1
    var delta = 0.0
    while (iter == 1 || delta > 0.00001) {
      val ctx = SampleContext(iter, iter)
      delta = (for {
        param ← allParams
      } yield {
        param.update(ctx)
      }).sum

      // DEBUGGING
      val params = newGlobal.toSeq
        .flatMap(_.parameters)
        .map {
          _.asInstanceOf[ParameterHolder[Id, Double]]
        }
      val mu = params(0).value.get
      val sigma = Floating.doubleFloating.exp(params(1).value.get)
      println(s"$iter $mu $sigma")

      iter = iter + 1
    }

    AEVBModel(newGlobal)
  }

  def distributionOf[U[_], S](variable: ContinuousVariable[U, S]): (U[Double], U[Double]) = ???
}

object AEVBModel {

  trait ParameterOptimizer {

    def initialize(target: Expression[Id, Double, Any], gc: GradientContext, ctx: SampleContext): Unit

    def update(ctx: SampleContext): Double
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
  case class ParameterHolder[U[_], S](parameter: Parameter[U, S], invFisher: Option[Expression[U, Double, S]] = None)
    extends ParameterOptimizer {

    private var valueEv: Evaluable[U[Double]] = null
    private var invFisherEv: Option[Evaluable[U[Double]]] = None
    private var gradientOptEv: Gradient[U, Id] = None

    // scale of the maximum update when inverse fisher is available
    private val scale = 5.0
    // mass of the heavy ball when using SGD
    private val mass = 0.9

    var value: Option[U[Double]] = None
    var momentum: Option[U[Double]] = None

    def initialize(target: Expression[Id, Double, Any], gc: GradientContext, ctx: SampleContext) = {
      valueEv = parameter.eval(gc)
      invFisherEv = invFisher.map { _.eval(gc) }
      value = Some(valueEv(ctx))
      gradientOptEv = target.grad(gc, parameter)
    }

    private def updateMomentum(gradValue: U[Double]): U[Double] = {
      val ops = parameter.ops
      val newMomentum = if (momentum.isDefined) {
        ops.zipMap(gradValue, momentum.get) {
          case (gv, pv) =>
            pv * mass + gv * (1.0 - mass)
        }
      } else {
        ops.map(gradValue) {
          _ * (1.0 - mass)
        }
      }
      momentum = Some(newMomentum)
      newMomentum
    }

    def update(ctx: SampleContext) = {
      value = Some(valueEv(ctx))

      val iF = invFisherEv.map { _(ctx) }

      gradientOptEv.map { gradEval ⇒
        gradEval(ctx)
      }.map { gradValue ⇒
        val rho = 1.0 / (ctx.iteration + 1)
        val vt = parameter.vt(ctx)
        val ops = parameter.ops

        // use inverse fisher matrix for natural gradient
        // fall back to SGD with momentum otherwise
        val deltaRaw = if (iF.isDefined) {
          val scaledGrad = ops.map(vt.tanh(
            ops.map(vt.times(gradValue, iF.get)){_ / scale}
          )){_ * scale}
          vt.times(scaledGrad, iF.get)
        } else {
          updateMomentum(gradValue)
        }

        // apply stochastic update with decreasing rate (longer history)
        // to make it more accurate
        val delta = ops.map(deltaRaw) {
          _ * rho
        }

        value = value.map { v ⇒
          vt.plus(v, delta)
        }

        parameter.v = value.get
        ops.foldLeft(vt.abs(delta))(0.0) { _ + _ }
      }.getOrElse(0.0)
    }
  }

  sealed trait VariablePosterior {
    // original factor - the one that's used in local samples
    def O: Factor

    // current approximating distribution - the prior for next batch of observations
    def P: Factor

    // target distribution - approximates the posterior
    def Q: Factor

    def parameters: Seq[ParameterOptimizer]

    def next: VariablePosterior
  }

  case class ContinuousVariablePosterior[U[_], S](original: Variable[U, Double, S],
                                                  variable: Variable[U, Double, S],
                                                  muStart: Option[Evaluable[U[Double]]] = None,
                                                  sigmaStart: Option[Evaluable[U[Double]]] = None)
    extends VariablePosterior {

    override val O: Variable[U, Double, S] = original

    override val P: Variable[U, Double, S] = variable

    implicit val fl = Floating.doubleFloating
    implicit val so = variable.so
    implicit val vOps: ContainerOps.Aux[U, S] = variable.ops

    val mu = new Parameter(muStart.getOrElse(variable.vt.map { vo ⇒ vo.zero }))
    val logSigma = new Parameter(sigmaStart.getOrElse(variable.vt.map { vo ⇒ vo.zero }))

    private val sigma = exp(logSigma)

    override val Q: Variable[U, Double, S] = Normal[U, S](mu, sigma).sample

    override val parameters = Seq(
        ParameterHolder(mu, Some(sigma)),
        ParameterHolder(logSigma, Some(Constant(mu.vt.map {
          vo => vo.div(vo.one, vo.sqrt(vo.fromInt(2)))
        })))
      )

    override def next = {
      ContinuousVariablePosterior(O, Q, Some(mu.value), Some(logSigma.value))
    }
  }

  case class CategoricalVariablePosterior(original: CategoricalVariable,
                                          variable: CategoricalVariable,
                                          thetasStart: Option[Evaluable[IndexedSeq[Double]]] = None)
    extends VariablePosterior {

    override val O: CategoricalVariable = original

    override val P: CategoricalVariable = variable

    private implicit val ops: ContainerOps.Aux[Id, Any] = variable.ops
    val thetas = new Parameter[IndexedSeq, Int](thetasStart.getOrElse(variable.K.map { k ⇒
      Array.fill(k)(1.0 / k): IndexedSeq[Double]
    }))

    override val Q: CategoricalVariable = Categorical(thetas).sample

    override val parameters = Seq(ParameterHolder(thetas))

    override def next = {
      CategoricalVariablePosterior(O, Q, Some(thetas.value))
    }
  }

  /**
   * Create a new model with a global set of variables to infer
   */
  def apply(global: Seq[Node]): AEVBModel = {

    // find new nodes, new variables & their (log) probability
    val collector = new VariableCollector(Set.empty, Map.empty)
    val (_, variables, parameters, factors, globalDeps) = collector.collect(global)

    // no optimizing hyper-parameters
    assert(parameters.isEmpty)

    val logP = variables.map {
      _.P.logp
    }.reduceOption(_ + _)
      .getOrElse(Constant(0.0))

    val logQ = variables.map {
      _.Q.logp
    }.reduceOption(_ + _)
      .getOrElse(Constant(0.0))

    val gc = new ModelGradientContext(
      variables.map { v ⇒
        (v.P: Node, v)
      }.toMap,
      Map.empty
    )
    val ctx = SampleContext(0, 0)

    val distParams = variables.flatMap(_.parameters)
    distParams.foreach(_.initialize(logP - logQ, gc, ctx))
    var iter = 1
    var delta = 0.0
    while (iter == 1 || delta > 0.00001 || iter < 1000) {
//    while (iter < 10000) {
      val ctx = SampleContext(iter, iter)
      delta = (for {
        param ← distParams
      } yield {
        param.update(ctx)
      }).sum

      // DEBUGGING
      val params = variables.toSeq
        .flatMap(_.parameters)
        .map {
          _.asInstanceOf[ParameterHolder[Id, Double]]
        }
      val mu = params(0).value.get
      val sigma = Floating.doubleFloating.exp(params(1).value.get)
      println(s"$iter $mu $sigma")

      iter = iter + 1
    }

    //    val globalSamplers = initialize(builders)
    AEVBModel(variables)
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
      val posterior = ContinuousVariablePosterior(v, v)
      variables += posterior
      deps += v -> stack.toSet
      collectVars(v.parents)
    }

    override def visitCategoricalVariable(v: CategoricalVariable): VariableCollector = {
      factors :+= v
      val posterior = CategoricalVariablePosterior(v, v)
      variables += posterior
      deps += v -> stack.toSet
      collectVars(v.parents)
    }

    override def visitNode(n: Node): VariableCollector =
      collectVars(n.parents)

  }

}

