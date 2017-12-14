package emmy.inference

import emmy.autodiff.{ Expression, Node, SampleContext }
import scalaz.Scalaz.Id

trait SamplerBuilder {

  def variable: Node

  def build(logP: Seq[Expression[Id, Double, Any]]): Sampler
}

trait SamplerInitializer extends SamplerBuilder {

  def eval(ec: SampleContext): Unit

}
