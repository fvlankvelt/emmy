package emmy.autodiff.inference

import emmy.autodiff.{TestVariable, Variable}
import emmy.inference.{AEVBSampler, ModelGradientContext, ModelSample}
import org.scalatest.FlatSpec

import scalaz.Scalaz.Id

class AEVBSamplerSpec extends FlatSpec {

  "The AEVB sampler" should "be close to the exact solution" in  {

    val variable = new TestVariable[Id, Double, Any](0.0)
    val logp = -(variable - 1.0) * (variable - 1.0) / 2.0

    val sampler = new AEVBSampler[Id, Double, Any](variable, 0.0, 1.0)
    val newSampler = (0 until 100).foldLeft(sampler) {
      case (s, _) =>
        val modelSample = new ModelSample[Double] {
          override def getSampleValue[U[_], S](n: Variable[U, Double, S]) =
            s.sample().asInstanceOf[U[Double]]
        }
        val gc = new ModelGradientContext[Double](modelSample)
        s.update(logp, gc, 0.05)._1
    }
    print(newSampler)
  }

}
