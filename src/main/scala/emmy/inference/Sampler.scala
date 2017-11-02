package emmy.inference

import emmy.autodiff.{ Expression, GradientContext, Node }
import scalaz.Scalaz.Id

trait Sampler {

  def variable: Node

  def update(logP: Expression[Id, Double, Any], gc: GradientContext, rho: Double): (Sampler, Double)

  def logp(): Expression[Id, Double, Any]
}
