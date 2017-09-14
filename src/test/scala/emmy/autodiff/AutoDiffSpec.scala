package emmy.autodiff

import emmy.autodiff.ContainerOps.Aux
import emmy.distribution.Normal
import emmy.inference._
import org.scalatest.FlatSpec

import scala.collection.mutable
import scala.util.Random
import scalaz.Scalaz._

class AutoDiffSpec extends FlatSpec {

  val gc = new GradientContext[Double] {

    private val cache = mutable.HashMap[AnyRef, Any]()

    override def apply[U[_], S](n: Expression[U, Double, S]): U[Double] = {
      n match {
        case v: TestVariable[U, Double, S] => v.value
        case v: Variable[U, Double, S] => cache.getOrElseUpdate(v, v.vt.rnd).asInstanceOf[U[Double]]
        case _ => n.apply(this)
      }
    }

    override def apply[W[_], U[_], T, S](n: Expression[U, Double, S], v: Variable[W, Double, T])(implicit wOps: Aux[W, T]): W[U[Double]] = {
      n.grad(this, v)
    }
  }

  val ec: EvaluationContext[Double] = gc

  case class TestVariable[U[_], V, S](value: U[V])
                                     (implicit
                                      val vo: ValueOps[U, V, S],
                                      val ops: ContainerOps.Aux[U, S])
    extends Variable[U, V, S] {

    override def shape = ops.shapeOf(value)

    override def apply(ec: EvaluationContext[V]) = value

    override def logp() = ???

    override implicit val vt = vo.bind(ops.shapeOf(value))
  }

  "AD" should "calculate scalar derivative" in {
    val x = TestVariable[Id, Double, Any](2.0)
    val y = x * x
    assert(y(ec) == 4.0)

    val z: Double = y.grad(gc, x)
    assert(z == 4.0)
  }

  it should "calculate vector derivative on List" in {
    val x = TestVariable[List, Double, Int](List(1.0, 2.0))
    val y = x * x
    assert(y(ec) == List(1.0, 4.0))

    val z = y.grad(gc, x)
    assert(z == List(List(2.0, 0.0), List(0.0, 4.0)))
  }

  it should "calculate derivative of a scalar function" in {
    val x = TestVariable[Id, Double, Any](2.0)
    val y = log(x)
    assert(y(ec) == scala.math.log(2.0))

    val z: Double = y.grad(gc, x)
    assert(z == 0.5)
  }

  it should "calculate derivative of a function applied to a list" in {
    val x = TestVariable[List, Double, Int](List(1.0, 2.0))
    val y = log(x)
    assert(y(ec) == List(0.0, scala.math.log(2.0)))

    val z = y.grad(gc, x)
    assert(z == List(List(1.0, 0.0), List(0.0, 0.5)))
  }

  it should "calculate probability of observation" in {
    val mu = TestVariable[List, Double, Int](List(0.0, 0.0))
    val sigma = TestVariable[List, Double, Int](List(1.0, 1.0))

    val normal = Normal(mu, sigma)
    val observation = normal.observe(List(1.0, 2.0))
    println(observation.logp()(ec))
  }

  it should "be able to implement linear regression" in {
    val a = Normal(0.0, 1.0).sample
    val b = Normal(List(0.0, 0.0), List(1.0, 1.0)).sample
    val e = Normal(1.0, 1.0).sample

    val data = List(
      (List(1.0, 2.0), 0.5),
      (List(2.0, 1.0), 1.0)
    )

    val observations = data.map {
      case (x, y) =>
        val s = a + sum(x * b)
        Normal(s, e).observe(y)
    }
    val logp = observations.map(_.logp()).sum +
      a.logp() + b.logp() + e.logp()
    println(logp(ec))

    val g_a: Double = logp.grad(gc, a)
    println(g_a)
  }

  it should "update variational parameters for each (minibatch of) data point(s)" in {
    val mu = Normal(0.0, 1.0).sample
    val sigma = Normal(0.0, 0.5).sample

    val initialModel = AEVBModel[Double](Seq(mu, sigma))
    val dist = Normal(mu, exp(sigma))

    var model = initialModel
    while(true) {
      val data = for {_ <- 0 until 100} yield {
        0.3 + Random.nextGaussian() * scala.math.exp(0.2)
      }

      val observations = data.map { d => dist.observe(d) }
      val newModel = model.update(observations)
      print(newModel)
      model = newModel
    }
  }

}
