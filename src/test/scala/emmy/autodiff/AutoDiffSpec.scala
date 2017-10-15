package emmy.autodiff

import emmy.TestVariable
import emmy.autodiff.ContainerOps.Aux
import emmy.distribution.Normal
import org.scalatest.FlatSpec

import scala.collection.mutable
import scalaz.Scalaz._

class AutoDiffSpec extends FlatSpec {

  val gc = new GradientContext {

    private val cache = mutable.HashMap[AnyRef, Any]()

    override def apply[U[_], V, S](n: Expression[U, V, S]): U[V] = {
      n match {
        case v: TestVariable[U, S] => v.value.asInstanceOf[U[V]]
        case v: Variable[U, S] => cache.getOrElseUpdate(v, v.vt(this).rnd).asInstanceOf[U[V]]
        case _ => n.apply(this)
      }
    }

    override def apply[W[_], U[_], V, T, S](n: Expression[U, V, S], v: Variable[W, T])(implicit wOps: Aux[W, T]): W[U[Double]] = {
      n.grad(this, v)
    }
  }

  val ec: EvaluationContext = gc


  "AD" should "calculate scalar derivative" in {
    val x = TestVariable(2.0)
    val y = x * x
    assert(y(ec) == 4.0)

    val z: Double = y.grad(gc, x)
    assert(z == 4.0)
  }

  it should "deal with constants" in {
    val x = TestVariable(0.0)
    val y = -(x - 1.0) * (x - 1.0) / 2.0
    assert(y(ec) == -0.5)

    val z: Double = y.grad(gc, x)
    assert(z == 1.0)
  }

  it should "calculate vector derivative on List" in {
    val x = TestVariable(List(1.0, 2.0))
    val y = x * x
    assert(y(ec) == List(1.0, 4.0))

    val z = y.grad(gc, x)
    assert(z == List(List(2.0, 0.0), List(0.0, 4.0)))
  }

  it should "calculate derivative of a reciprocal" in {
    val x = TestVariable(2.0)
    val y = Constant(1.0) / x
    assert(y(ec) == 0.5)

    val z: Double = y.grad(gc, x)
    assert(z == -0.25)
  }

  it should "calculate derivative of a scalar function" in {
    val x = TestVariable(2.0)
    val y = log(x)
    assert(y(ec) == scala.math.log(2.0))

    val z: Double = y.grad(gc, x)
    assert(z == 0.5)
  }

  it should "calculate derivative of a function applied to a list" in {
    val x = TestVariable(List(1.0, 2.0))
    val y = log(x)
    assert(y(ec) == List(0.0, scala.math.log(2.0)))

    val z = y.grad(gc, x)
    assert(z == List(List(1.0, 0.0), List(0.0, 0.5)))
  }

  it should "calculate probability of observation" in {
    val mu = TestVariable(List(0.0, 0.0))
    val sigma = TestVariable(List(1.0, 1.0))

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

  it should "calculate exp derivative" in {
    val x = TestVariable(1.0)
    val y = exp(x)
    assert(y(ec) == scala.math.exp(1.0))

    val z: Double = y.grad(gc, x)
    assert(z == scala.math.exp(1.0))
  }

  it should "derive zero for gradient of constant" in {
    val x = TestVariable(1.0)
    val y = Constant(2.0)
    val g: Double = y.grad(gc, x)
    assert(g == 0.0)
  }
}
