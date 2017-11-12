package emmy.inference.aevb

import breeze.numerics.abs
import emmy.autodiff.{ CategoricalVariable, Constant, ConstantLike, ContinuousVariable, EvaluationContext, Expression, GradientContext }
import emmy.distribution.CategoricalFactor
import emmy.inference.Sampler

import scala.util.Random
import scalaz.Scalaz.Id

trait SamplerVariable[U[_], T]
  extends ContinuousVariable[U, T] with ConstantLike[U, Double, T] {

  override val value: Expression[U, Double, T]
}

/**
 * Use a Categorical distribution as variational approximation to a Categorical variable.
 */
class CategoricalSampler(
    val variable: CategoricalVariable,
    val thetas:   IndexedSeq[Double],
    val offset:   Double              = 0.0
) extends Sampler {

  private val Q = {
    val v = variable
    val t = thetas
    new CategoricalFactor {

      override val variable = v

      override val thetas = Constant(t)
    }
  }

  //@formatter:off
  /**
   * note: logP is not normalized - so deltaLog has non-zero offset
   * The expectation of the gradient of this constant is zero, however:
   * E_Q[grad(log Q) * cst] =
   *   cst * E_Q[grad(log Q)] =
   *       cst * grad(E_Q[1]) = 0
   * The constant is tracked to speed up convergence - still needs to be verified though
   */
  //@formatter:on
  def update(logP: Expression[Id, Double, Any], eval: GradientContext, rho: Double): (CategoricalSampler, Double) = {
    val index = eval(variable)
    val gradLogQ = gradLogTheta(index)
    val lp = eval(logP)
    val newOffset = (1.0 - rho) * offset + rho * lp

    val deltaLog = eval(logp()) - (lp - newOffset)
    val newThetas = (thetas zip gradLogQ).map {
      case (theta, grad) ⇒
        val dLogTheta = -grad * rho * deltaLog
        val factor = if (dLogTheta > 0) 1 + dLogTheta else 1.0 / (1 - dLogTheta)
        theta * factor
    }
    val sum = newThetas.sum
    val delta = (thetas zip gradLogQ).map {
      case (theta, grad) ⇒
        theta * abs(grad)
    }.sum * rho * abs(deltaLog)
    (
      new CategoricalSampler(
        variable, newThetas.map {
        _ / sum
      },
        newOffset
      ),
      delta
    )
    //    (new CategoricalSampler(variable, newThetas, offset), delta)
  }

  private def gradLogTheta(index: Int): IndexedSeq[Double] = {
    val thetaSum = thetas.sum
    thetas.zipWithIndex.map {
      case (theta, idx) if idx == index ⇒
        1.0 - theta / thetaSum
      case (theta, _) ⇒
        -theta / thetaSum
    }
  }

  /**
   * Log probability of the variable, to be used as the prior distribution in a streaming variational approximation.
   */
  def logp(): Expression[Id, Double, Any] = {
    Q.logp()
  }

  // draw from P?
  def sample(ec: EvaluationContext): Int = {
    val sumThetas = thetas.sum
    var draw = Random.nextDouble() * sumThetas
    for { (theta, idx) ← thetas.zipWithIndex } {
      if (theta > draw) {
        return idx
      }
      draw -= theta
    }
    throw new UnsupportedOperationException("Uniform draw is larger than 1.0")
  }
}
