package emmy.inference

import breeze.numerics.abs
import emmy.TestVariable
import emmy.autodiff.Node
import emmy.distribution.Normal
import org.scalatest.FlatSpec

import scalaz.Scalaz.Id

class AEVBSamplerSpec extends FlatSpec {

  "The AEVB sampler" should "determine \\mu close to the exact solution" in  {

    val variable = new TestVariable[Id, Any](0.0)
    val logp = -(variable - 1.0) * (variable - 1.0) / 2.0

    val sampler = new AEVBSampler[Id, Any](variable, 0.0, 1.0)
    val newSampler = (0 until 200).foldLeft(sampler) {
      case (s, _) =>
        val model = new AEVBSamplersModel(Map((variable: Node) -> s))
        val gc = new ModelGradientContext(model)
        s.update(logp, gc, 1.0)._1
    }
    assert(abs(newSampler.mu - 1.0) < 0.01)
  }

  it should "determine \\sigma close to the exact solution" in {

    val variable = new TestVariable[Id, Any](0.0)
    val logp = -variable * variable / 8.0

    val sampler = new AEVBSampler[Id, Any](variable, 0.0, 0.5)
    val newSampler = (0 until 400).foldLeft(sampler) {
      case (s, _) =>
        val model = new AEVBSamplersModel(Map((variable: Node) -> s))
        val gc = new ModelGradientContext(model)
        s.update(logp, gc, 1.0)._1
    }
    assert(abs(newSampler.sigma - 2.0) < 0.01)
  }

  it should "reconstruct normal sample parameters" in {
    val variable = Normal[Id, Any](1.0, 2.0).sample
    val logp = variable.logp()

    val sampler = new AEVBSampler[Id, Any](variable, 0.0, 0.5)
    val newSampler = (0 until 200).foldLeft(sampler) {
      case (s, _) =>
        val model = new AEVBSamplersModel(Map((variable: Node) -> s))
        val gc = new ModelGradientContext(model)
        s.update(logp, gc, 1.0)._1
    }
    assert(abs(newSampler.mu - 1.0) < 0.01)
    assert(abs(newSampler.sigma - 2.0) < 0.01)
  }
}
