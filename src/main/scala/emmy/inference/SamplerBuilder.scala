package emmy.inference

import emmy.autodiff.{ EvaluationContext, Node }

trait SamplerBuilder {

  def variable: Node

  def eval(ec: EvaluationContext): Unit

  def build(): Sampler
}
