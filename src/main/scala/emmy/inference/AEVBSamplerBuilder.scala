package emmy.inference

import emmy.autodiff.{ContinuousVariable, EvaluationContext, ValueOps, Variable}

import scala.collection.mutable

case class AEVBSamplerBuilder[U[_], S](variable: ContinuousVariable[U, S]) {
  private val samples: mutable.Buffer[U[Double]] = mutable.Buffer.empty
  private var numUVOpt: Option[ValueOps[U, Double, S]] = None

  def eval(ec: EvaluationContext): Unit = {
    samples += ec(variable)
    numUVOpt match {
      case Some(numV) =>
        assert(numV == variable.vt(ec))
      case None =>
        numUVOpt = Some(variable.vt(ec))
    }
  }

  def build(): AEVBSampler[U, S] = {
    implicit val numUV = numUVOpt.get
    val size = samples.length
    val mu: U[Double] = numUV.div(samples.sum(numUV), numUV.fromInt(size))
    val sigma2 = samples.map { x =>
      val delta = numUV.minus(x, mu)
      numUV.times(delta, delta)
    }.sum(numUV)
    val ratio = numUV.div(sigma2, numUV.fromInt(size - 1))
    val sigma = numUV.sqrt(ratio)
    new AEVBSampler(variable, mu, sigma)
  }
}
