package emmy.inference

import emmy.autodiff.{ Expression, GradientContext, Node }
import emmy.distribution.Factor

import scalaz.Scalaz.Id

trait Sampler extends Factor {

  def variable: Node

  def update(logP: Seq[Expression[Id, Double, Any]], gc: GradientContext, rho: Double): (Sampler, Double)
}
