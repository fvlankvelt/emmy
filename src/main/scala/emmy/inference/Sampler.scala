package emmy.inference

import emmy.autodiff.{ GradientContext, Node, SampleContext }
import emmy.distribution.Factor

trait Sampler extends Factor {

  def variable: Node

  def update(sc: SampleContext, rho: Double): Double

  def toBuilder(ec: GradientContext): SamplerBuilder
}

