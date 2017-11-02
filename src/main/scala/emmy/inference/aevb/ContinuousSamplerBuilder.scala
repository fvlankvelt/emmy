package emmy.inference.aevb

import emmy.autodiff.{ ContinuousVariable, EvaluationContext, ValueOps }
import emmy.inference.SamplerBuilder

import scala.collection.mutable

case class ContinuousSamplerBuilder[U[_], S](variable: ContinuousVariable[U, S]) extends SamplerBuilder {
  private val samples: mutable.Buffer[U[Double]] = mutable.Buffer.empty
  private var numUVOpt: Option[ValueOps[U, Double, S]] = None

  def eval(ec: EvaluationContext): Unit = {
    samples += ec(variable)
    numUVOpt match {
      case Some(numV) ⇒
        assert(numV == variable.vt(ec))
      case None ⇒
        numUVOpt = Some(variable.vt(ec))
    }
  }

  def build(): ContinuousSampler[U, S] = {
    implicit val numUV = numUVOpt.get
    val size = samples.length
    val mu: U[Double] = numUV.div(samples.sum(numUV), numUV.fromInt(size))
    val sigma2 = samples.map { x ⇒
      val delta = numUV.minus(x, mu)
      numUV.times(delta, delta)
    }.sum(numUV)
    val ratio = numUV.div(sigma2, numUV.fromInt(size - 1))
    val sigma = numUV.sqrt(ratio)
    new ContinuousSampler(variable, mu, sigma)
  }
}
