package emmy.inference.aevb

import emmy.autodiff._

import scalaz.Scalaz.Id

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
    invFisherEv = invFisher.map {
      _.eval(gc)
    }
    value = Some(valueEv(ctx))
    gradientOptEv = target.grad(gc, parameter)
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

  def update(ctx: SampleContext) = {
    value = Some(valueEv(ctx))

    val iF = invFisherEv.map {
      _(ctx)
    }

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
          ops.map(vt.times(gradValue, iF.get)) {
            _ / scale
          }
        )) {
          _ * scale
        }
        vt.times(scaledGrad, iF.get)
      }
      else {
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
      ops.foldLeft(vt.abs(delta))(0.0) {
        _ + _
      }
    }.getOrElse(0.0)
  }
}
