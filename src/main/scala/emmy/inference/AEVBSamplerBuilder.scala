package emmy.inference

import emmy.autodiff.{EvaluationContext, ValueOps, Variable}
import scalaz.Scalaz.Id

import scala.collection.mutable

case class AEVBSamplerBuilder[U[_], V, S](variable: Variable[U, V, S]) {
  private val samples: mutable.Buffer[U[V]] = mutable.Buffer.empty

  def eval(ec: EvaluationContext[V]): Unit = {
    samples += ec(variable)
  }

  def build()(implicit idT: ValueOps[Id, V, Any]): AEVBSampler[U, V, S] = {
    implicit val numUV = variable.vt
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
