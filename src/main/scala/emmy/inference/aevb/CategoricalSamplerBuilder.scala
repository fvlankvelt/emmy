package emmy.inference.aevb

import emmy.autodiff.{ CategoricalVariable, EvaluationContext }
import emmy.inference.SamplerBuilder

import scala.collection.mutable

case class CategoricalSamplerBuilder(variable: CategoricalVariable) extends SamplerBuilder {
  private val samples: mutable.Buffer[Int] = mutable.Buffer.empty
  private var kOpt: Option[Int] = None

  def eval(ec: EvaluationContext): Unit = {
    samples += ec(variable)
    kOpt match {
      case Some(k) ⇒
        assert(k == variable.K(ec))
      case None ⇒
        kOpt = Some(variable.K(ec))
    }
  }

  def build(): CategoricalSampler = {
    val k = kOpt.get
    val counts = Array.fill(k)(1)
    for (sample ← samples) {
      counts(sample) += 1
    }
    val total = counts.sum
    new CategoricalSampler(variable, counts.map { c ⇒ c * 1.0 / total })
  }
}
