package emmy.inference.aevb

import emmy.autodiff.{ Expression, GradientContext, SampleContext }

import scalaz.Scalaz.Id

trait ParameterOptimizer {

  def initialize(target: Expression[Id, Double, Any], gc: GradientContext, ctx: SampleContext): Unit

  def update(ctx: SampleContext): Double
}
