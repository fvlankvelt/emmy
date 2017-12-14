package emmy.inference.aevb

import breeze.linalg.DenseVector
import breeze.numerics.abs
import emmy.TestVariable
import emmy.autodiff._
import emmy.distribution._
import emmy.inference.aevb.AEVBModel.{ ParameterHolder, ParameterOptimizer }
import emmy.inference.{ Model, ModelGradientContext, aevb }
import org.scalatest.FlatSpec

import scala.util.Random
import scalaz.Scalaz.Id
//
//class TestNode(override val parents: Seq[Node]) extends Node {
//  override def toString = s"node(${parents.mkString(",")})"
//}
//
//object TestNode {
//  def apply(parents: Node*) = new TestNode(parents.toSeq)
//}
//
class AEVBModelSpec extends FlatSpec {

  "The AEVB model" should "initialize" in {
    val dist = Normal(0.5, 0.7).sample
    val model = AEVBModel(Seq[Node](dist))
    val params = model.parameters.map {
      _.asInstanceOf[ParameterHolder[Id, Double]]
    }
    val es = math.log(0.7)
    assert(abs(params(0).value.get - 0.5) < 0.1)
    assert(abs(params(1).value.get - math.log(0.7)) < 0.05)
  }

  //
  //  "The AEVB model" should "collect node graph" in {
  //    val variable = TestVariable(1.0)
  //    val graph = TestNode(
  //      TestNode(
  //        variable
  //      ),
  //      TestNode(
  //        variable
  //      )
  //    )
  //    val gc = new ModelGradientContext(GlobalModel, Map.empty)
  //    val collector = new aevb.AEVBModel.VariableCollector(Set.empty, Map.empty, gc)
  //    val (nodes, _, _, deps) = collector.collect(Seq(graph))
  //    assert(nodes.size == 4)
  //    assert(deps.size == 1)
  //    assert(deps(variable).size == 4)
  //  }
  //
  //  it should "update mu for each (minibatch of) data point(s)" in {
  //    val mu = Normal(0.0, 1.0).sample
  //
  //    val initialModel = AEVBModel(Seq(mu))
  //    val dist = Normal(mu, 0.2)
  //
  //    val finalModel = (0 until 100).foldLeft(initialModel) {
  //      case (model, _) ⇒
  //        val data = for { _ ← 0 until 100 } yield {
  //          0.3 + Random.nextGaussian() * 0.2
  //        }
  //
  //        val observations = data.map { d ⇒ dist.observe(d) }
  //        model.update(observations)
  //    }
  //    val sampler = finalModel.getSampler[Id, Double, Any](mu)
  //    assert(abs(sampler.mu - 0.3) < 0.01)
  //  }
  //
  //  it should "update sigma for each (minibatch of) data point(s)" in {
  //    val logSigma = Normal(0.5, 1.0).sample
  //
  //    val initialModel = AEVBModel(Seq(logSigma))
  //    val dist = Normal(0.3, exp(logSigma))
  //
  //    val finalModel = (0 until 100).foldLeft(initialModel) {
  //      case (model, _) ⇒
  //        val data = for { _ ← 0 until 100 } yield {
  //          0.3 + Random.nextGaussian() * scala.math.exp(0.2)
  //        }
  //
  //        val observations = data.map { d ⇒ dist.observe(d) }
  //        model.update(observations)
  //    }
  //    val sampler = finalModel.getSampler[Id, Double, Any](logSigma)
  //    assert(abs(sampler.mu - 0.2) < 0.05)
  //  }
  //
  //  def printVariable[U[_], V, S](model: AEVBModel, name: String, variable: ContinuousVariable[U, S]): Unit = {
  //    val dist = model.distributionOf(variable)
  //    println(s"$name: mu = ${dist._1}, sigma = ${dist._2}")
  //  }
  //
  //  it should "be able to implement linear regression" in {
  //    // data generation - to be reproduced
  //    val data = {
  //      val alpha = 1.5
  //      val sigma = 1.0
  //      val beta = List(1.0, 2.5)
  //
  //      (for { _ ← 0 until 200 } yield {
  //        val X = List(Random.nextGaussian(), 0.2 * Random.nextGaussian())
  //        val Y = alpha + X(0) * beta(0) + X(1) * beta(1) + Random.nextGaussian() * sigma
  //        (X, Y)
  //      }).toList
  //    }
  //
  //    // model parameters with their priors
  //    val a = Normal(0.0, 1.0).sample
  //    val b = Normal(List(0.0, 0.0), List(1.0, 1.0)).sample
  //    val e = Normal(0.0, 1.0).sample
  //    val model = AEVBModel(Seq[Node](a, b, e))
  //
  //    println("Prior model:")
  //    printVariable(model, "a", a)
  //    printVariable(model, "b", b)
  //    printVariable(model, "e", e)
  //
  //    // infer parameter values from observations
  //    val observations = data.map {
  //      case (x, y) ⇒
  //        val s = a + sum(x * b)
  //        Normal(s, e).observe(y)
  //    }
  //    val newModel = model.update(observations)
  //    println("Posterior model:")
  //    printVariable(newModel, "a", a)
  //    printVariable(newModel, "b", b)
  //    printVariable(newModel, "e", e)
  //
  //  }
  //
  //  it should "do categorical regression" in {
  //    val data = () ⇒ {
  //      val x = 1.5
  //      val p = 1.0 / (1.0 + math.exp(x))
  //      val vec: DenseVector[Double] = DenseVector(p, 1.0 - p)
  //      val dist = breeze.stats.distributions.Multinomial(vec)
  //      dist.sample(100).toList
  //    }
  //    val testvar = Normal(0.0, 1.0).sample
  //    val pvar = 1.0 / (1.0 + exp(testvar))
  //    val multi = Categorical(Vector(pvar, 1.0 - pvar))
  //
  //    var model = AEVBModel(Seq[Node](testvar))
  //    println("Prior model:")
  //    printVariable(model, "testvar", testvar)
  //
  //    for { _ ← 0 until 100 } {
  //      val observations = data().map { values ⇒ multi.observe(values) }
  //      val newModel = model.update(observations)
  //      model = newModel
  //    }
  //    println("Posterior model:")
  //    printVariable(model, "testvar", testvar)
  //
  //    val sampler = model.getSampler[Id, Double, Any](testvar)
  //    assert(abs(sampler.mu - 1.5) < 0.10)
  //  }
  //
  //  it should "infer mixture models" in {
  //    val data = () ⇒ {
  //      val mu = Seq(-1.0, 1.0)
  //      val sigma = Seq(0.5, 0.5)
  //      val dists = (mu zip sigma).map {
  //        case (m, s) ⇒
  //          breeze.stats.distributions.Gaussian(m, s)
  //      }
  //      val p = 1.0 / (1.0 + math.exp(0.2))
  //      val vec: DenseVector[Double] = DenseVector(p, 1.0 - p)
  //      val dist = breeze.stats.distributions.Multinomial(vec)
  //      dist.sample(100).map { idx ⇒
  //        dists(idx).sample()
  //      }
  //    }
  //
  //    val activation = Normal(0.0, 1.0).sample
  //    val pvar = 1.0 / (1.0 + exp(activation))
  //    val multi = Categorical(Vector(pvar, 1.0 - pvar))
  //
  //    val mus = Range(0, 2).map(_ ⇒ Normal(0.0, 0.5).sample)
  //    val clusters = mus.map { m ⇒ Normal(m, 0.5) }
  //    val result = Select(multi, clusters)
  //
  //    var model = AEVBModel((mus: Seq[Node]) :+ activation)
  //    println("Prior model:")
  //
  //    for { _ ← 0 until 10 } {
  //      val d = data()
  //      val observations = d.map { x ⇒ result.observe(x) }
  //      val newModel = model.update(observations)
  //      model = newModel
  //      println("Posterior model:")
  //      printVariable(model, "activation", activation)
  //      printVariable(model, "mu(0)", mus(0))
  //      printVariable(model, "mu(1)", mus(1))
  //    }
  //    {
  //      val dists = mus.map { mu ⇒ model.distributionOf(mu) }
  //      assert(abs(dists(0)._1 + dists(1)._1) < 0.15)
  //      assert(abs(abs(dists(0)._1 - dists(1)._1) - 2.0) < 0.3)
  //    }
  //  }
  //
}
