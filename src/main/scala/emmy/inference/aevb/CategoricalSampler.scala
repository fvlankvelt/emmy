package emmy.inference.aevb

import emmy.autodiff.{ CategoricalVariable, Constant, EvaluationContext, Expression, GradientContext }
import emmy.distribution.CategoricalSample
import emmy.inference.Sampler

import scalaz.Scalaz.Id

/**
 * Use a Categorical distribution as variational approximation to a Categorical variable.
 */
class CategoricalSampler(
    val variable: CategoricalVariable,
    val thetas:   IndexedSeq[Double]
) extends Sampler {

  private val thetaVar = new SamplerVariable[IndexedSeq, Int] {

    override val value = Constant[IndexedSeq, Double, Int](thetas)

    override def logp() = ???
  }
  private val Q = new CategoricalSample(thetaVar)

  def update(logP: Expression[Id, Double, Any], gc: GradientContext, rho: Double): (CategoricalSampler, Double) = {
    val gradLogQ = gc(Q, thetaVar)
    val deltaLog = gc(logp() - logP)
    val newThetas = (thetas zip gradLogQ).map {
      case (theta, grad) ⇒ (1.0 - rho) * theta + rho * grad * deltaLog
    }
    (new CategoricalSampler(variable, newThetas), 0.0)
  }

  /**
   * Log probability of the variable, to be used as the prior distribution in a streaming variational approximation.
   */
  def logp(): Expression[Id, Double, Any] = {
    Q.logp()
  }

  def sample(ec: EvaluationContext): Int = {
    Q(ec)
  }
}
