package emmy.inference

import emmy.autodiff.{EvaluationContext, Floating, ValueOps, Variable}

import scala.collection.mutable

case class AEVBSamplerBuilder[U[_], V, S](variable: Variable[U, V, S]) {
  private val samples: mutable.Buffer[U[V]] = mutable.Buffer.empty
  private var numUVOpt: Option[ValueOps[U, V, S]] = None

  def eval(ec: EvaluationContext[V]): Unit = {
    samples += ec(variable)
    numUVOpt match {
      case Some(numV) =>
        assert(numV == variable.vt(ec))
      case None =>
        numUVOpt = Some(variable.vt(ec))
    }
  }

  def build()(implicit fl: Floating[V]): AEVBSampler[U, V, S] = {
    implicit val numUV = numUVOpt.get
    val size = samples.length
    val mu: U[V] = numUV.div(samples.sum(numUV), numUV.fromInt(size))
    val sigma2 = samples.map { x =>
      val delta = numUV.minus(x, mu)
      numUV.times(delta, delta)
    }.sum(numUV)
    val ratio = numUV.div(sigma2, numUV.fromInt(size - 1))
    val sigma = numUV.sqrt(ratio)
    new AEVBSampler(variable, mu, sigma)
  }
}
