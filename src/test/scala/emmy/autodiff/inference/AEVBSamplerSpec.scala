package emmy.autodiff.inference

import breeze.numerics.abs
import emmy.autodiff.{TestVariable, Variable}
import emmy.distribution.NormalSample
import emmy.inference.{AEVBSampler, ModelGradientContext, ModelSample}
import org.scalatest.FlatSpec

import scalaz.Scalaz.Id

class AEVBSamplerSpec extends FlatSpec {

  "The AEVB sampler" should "determine \\mu close to the exact solution" in  {

    val variable = new TestVariable[Id, Double, Any](0.0)
    val logp = -(variable - 1.0) * (variable - 1.0) / 2.0

    val sampler = new AEVBSampler[Id, Double, Any](variable, 0.0, 1.0)
    val newSampler = (0 until 200).foldLeft(sampler) {
      case (s, _) =>
        val modelSample = new ModelSample[Double] {
          override def getSampleValue[U[_], S](n: Variable[U, Double, S]) =
            s.sample().asInstanceOf[U[Double]]
        }
        val gc = new ModelGradientContext[Double](modelSample)
        s.update(logp, gc, 1.0)._1
    }
    assert(abs(newSampler.mu - 1.0) < 0.01)
  }

  it should "determine \\sigma close to the exact solution" in {

    val variable = new TestVariable[Id, Double, Any](0.0)
    val logp = -variable * variable / 8.0

    val sampler = new AEVBSampler[Id, Double, Any](variable, 0.0, 0.5)
    val newSampler = (0 until 200).foldLeft(sampler) {
      case (s, _) =>
        val modelSample = new ModelSample[Double] {
          override def getSampleValue[U[_], S](n: Variable[U, Double, S]) =
            s.sample().asInstanceOf[U[Double]]
        }
        val gc = new ModelGradientContext[Double](modelSample)
        s.update(logp, gc, 1.0)._1
    }
    assert(abs(newSampler.sigma - 2.0) < 0.01)
  }


  it should "reconstruct normal stochast parameters" in {
    val variable = new NormalSample[Id, Double, Any](1.0, 2.0)
    val logp = variable.logp()

    val sampler = new AEVBSampler[Id, Double, Any](variable, 0.0, 0.5)
    val newSampler = (0 until 200).foldLeft(sampler) {
      case (s, _) =>
        val modelSample = new ModelSample[Double] {
          override def getSampleValue[U[_], S](n: Variable[U, Double, S]) =
            s.sample().asInstanceOf[U[Double]]
        }
        val gc = new ModelGradientContext[Double](modelSample)
        s.update(logp, gc, 1.0)._1
    }
    assert(abs(newSampler.mu - 1.0) < 0.01)
    assert(abs(newSampler.sigma - 2.0) < 0.01)
  }

}
