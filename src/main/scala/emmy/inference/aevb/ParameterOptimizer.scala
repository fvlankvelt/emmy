package emmy.inference.aevb

import emmy.autodiff._

import scalaz.Scalaz
import scalaz.Scalaz.Id

trait ParameterOptimizer {

  def initialize(
    logp: Expression[Id, Double, Any],
    logq: Expression[Id, Double, Any],
    gc:   GradientContext,
    ctx:  SampleContext
  ): Unit

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
trait GradientBasedOptimizer[U[_], S] extends ParameterOptimizer {

  def parameter: Parameter[U, S]

  var value: Option[U[Double]] = None

  // mass of the heavy ball when using SGD
  protected val mass = 0.9

  protected var valueEv: Evaluable[U[Double]] = null
  private var gradientOptEv: Gradient[U, Id] = None

  private var momentum: Option[U[Double]] = None

  final override def initialize(logp: Expression[Scalaz.Id, Double, Any], logq: Expression[Scalaz.Id, Double, Any], gc: GradientContext, ctx: SampleContext): Unit = {
    valueEv = parameter.eval(gc)
    value = Some(valueEv(ctx))
    gradientOptEv = calculateGradient(logp, logq, gc)
  }

  protected def calculateGradient(logp: Expression[Scalaz.Id, Double, Any], logq: Expression[Scalaz.Id, Double, Any], gc: GradientContext): Gradient[U, Id] = None

  final override def update(ctx: SampleContext) = {
    value = Some(valueEv(ctx))

    gradientOptEv.map { gradEval ⇒
      gradEval(ctx)
    }.map { gradValue ⇒
      val rho = 1.0 / (ctx.iteration + 1)
      val vt = parameter.vt(ctx)
      val ops = parameter.ops

      // use inverse fisher matrix for natural gradient
      // fall back to SGD with momentum otherwise
      val deltaRaw = updateMomentum(gradValue)

      // apply stochastic update with decreasing rate (longer history)
      // to make it more accurate
      val delta = ops.map(deltaRaw) {
        _ * rho
      }

      value = value.map { v ⇒
        vt.plus(v, delta)
      }

      parameter.v = value.get
      ops.foldLeft(vt.abs(delta))(0.0) {
        _ + _
      }
    }.getOrElse(0.0)
  }

  private def updateMomentum(gradValue: U[Double]): U[Double] = {
    val ops = parameter.ops
    val newMomentum = if (momentum.isDefined) {
      ops.zipMap(gradValue, momentum.get) {
        case (gv, pv) ⇒
          pv * mass + gv * (1.0 - mass)
      }
    }
    else {
      ops.map(gradValue) {
        _ * (1.0 - mass)
      }
    }
    momentum = Some(newMomentum)
    newMomentum
  }

}

case class ReparameterizedOptimizer[U[_], S](
    parameter: Parameter[U, S],
    invFisher: Expression[U, Double, S]
)
  extends GradientBasedOptimizer[U, S] {

  // scale of the maximum update when inverse fisher is available
  private val scale = 5.0

  /**
   * Gradient only depends on parameter via variable
   */
  override protected def calculateGradient(logp: Expression[Scalaz.Id, Double, Any], logq: Expression[Scalaz.Id, Double, Any], gc: GradientContext): Gradient[U, Scalaz.Id] = {
    val invFisherEv = invFisher.eval(gc)
    val delta = logp - logq
    val g = delta.grad(gc, parameter)
    g.map { gv ⇒ ctx ⇒
      val iF = invFisherEv(ctx)
      val vt = parameter.vt(ctx)
      val ops = parameter.ops

      val scaledGrad = ops.map(vt.tanh(
        ops.map(vt.times(gv(ctx), iF)) {
          _ / scale
        }
      )) {
        _ * scale
      }
      vt.times(scaledGrad, iF)
    }
  }

}

case class ScoreFunctionOptimizer[U[_], S](parameter: Parameter[U, S])
  extends GradientBasedOptimizer[U, S] {

  override protected def calculateGradient(logp: Expression[Scalaz.Id, Double, Any], logq: Expression[Scalaz.Id, Double, Any], gc: GradientContext): Gradient[U, Scalaz.Id] = {
    val deltaExpr = logp - logq
    val deltaEv = deltaExpr.eval(gc)
    val scoreGradOpt = logq.grad(gc, parameter)

    scoreGradOpt.map { scoreGradEv ⇒ ctx ⇒
      val scoreGrad = scoreGradEv(ctx)
      val delta = deltaEv(ctx)
      parameter.ops.map(scoreGrad) { _ * delta }
    }
  }

}

case class FunctionOptimizer[U[_], S](parameter: Parameter[U, S])
  extends GradientBasedOptimizer[U, S] {

  override protected def calculateGradient(target: Expression[Scalaz.Id, Double, Any], ignored: Expression[Scalaz.Id, Double, Any], gc: GradientContext): Gradient[U, Scalaz.Id] = {
    target.grad(gc, parameter)
  }

}
