package emmy.inference.aevb

import breeze.numerics.abs
import emmy.autodiff.{Evaluable, Parameter, SampleContext}
import emmy.inference.ModelGradientContext
import org.scalatest.FlatSpec

import scalaz.Scalaz.Id

class ParameterHolderSpec extends FlatSpec {

  "The Parameter holder" should "determine the value close to the exact solution" in {
    val variable = new Parameter[Id, Any](Evaluable.fromConstant(0.0))
    val logp = -(variable - 1.0) * (variable - 1.0) / 2.0

    val sampler = ParameterHolder(variable)
    val gc = new ModelGradientContext(Map.empty)
    val ctx = SampleContext(0, 0)
    sampler.initialize(logp, gc, ctx)
    for { iter <- Range(0, 100) } {
      sampler.update(SampleContext(iter, iter))
    }
    assert(abs(sampler.value.get - 1.0) < 0.001)
  }

}
