package emmy.autodiff.inference

import breeze.numerics.abs
import emmy.autodiff.exp
import emmy.distribution.Normal
import emmy.inference.{AEVBModel, AEVBSampler}
import org.scalatest.FlatSpec

import scala.util.Random
import scalaz.Scalaz.Id

class AEVBModelSpec extends FlatSpec {

  "The AEVB model" should "update mu for each (minibatch of) data point(s)" in {
    val mu = Normal(0.0, 1.0).sample

    val initialModel = AEVBModel[Double](Seq(mu))
    val dist = Normal(mu, 0.2)

    val finalModel = (0 until 10).foldLeft(initialModel) {
      case (model, _) =>
        val data = for {_ <- 0 until 100} yield {
          0.3 + Random.nextGaussian() * 0.2
        }

        val observations = data.map { d => dist.observe(d) }
        model.update(observations)
    }
    val sampler = finalModel.globalVars.head._2.asInstanceOf[AEVBSampler[Id, Double, Any]]
    assert(abs(sampler.mu - 0.3) < 0.01)
  }

  it should "update sigma for each (minibatch of) data point(s)" in {
    val logSigma = Normal(0.5, 1.0).sample

    val initialModel = AEVBModel[Double](Seq(logSigma))
    val dist = Normal(0.3, exp(logSigma))

    val finalModel = (0 until 100).foldLeft(initialModel) {
      case (model, _) =>
        val data = for {_ <- 0 until 100} yield {
          0.3 + Random.nextGaussian() * scala.math.exp(0.2)
        }

        val observations = data.map { d => dist.observe(d) }
        model.update(observations)
    }
    val sampler = finalModel.globalVars.head._2.asInstanceOf[AEVBSampler[Id, Double, Any]]
    assert(abs(sampler.mu - 0.2) < 0.05)
  }

}
