package emmy.autodiff

import emmy.TestVariable
import emmy.distribution.Normal
import org.scalatest.FlatSpec
import scalaz.Scalaz.Id

import scala.collection.mutable

class AutoDiffSpec extends FlatSpec {

  private val gc = new GradientContext {

    private val cache = mutable.HashMap[AnyRef, Any]()

    override def apply[U[_], V, S](n: Expression[U, V, S]): Evaluable[U[V]] = {
      n match {
        case v: TestVariable[U, S]       ⇒
          v.value.asInstanceOf[U[V]]

        case v: ContinuousVariable[U, S] ⇒
          cache.getOrElseUpdate(v, v.vt.map(_.rnd)).asInstanceOf[Evaluable[U[V]]]

        case _                           ⇒
          n.eval(this)
      }
    }

    override def apply[W[_], U[_], V, T, S](n: Expression[U, V, S], v: Parameter[W, T]): Option[Evaluable[W[U[Double]]]] = {
      n.grad(this, v)
    }
  }

  private val sc: SampleContext = SampleContext(0, 0)

  "AD" should "calculate scalar derivative" in {
    val x = Parameter[Id, Any](2.0)
    val y = x * x
    assert(y.eval(gc)(sc) == 4.0)

    val z: Double = y.grad(gc, x).get(sc)
    assert(z == 4.0)
  }

  it should "deal with constants" in {
    val x = Parameter[Id, Any](0.0)
    val y = -(x - 1.0) * (x - 1.0) / 2.0
    assert(y.eval(gc)(sc) == -0.5)

    val z: Double = y.grad(gc, x).get(sc)
    assert(z == 1.0)
  }

  it should "calculate vector derivative on List" in {
    val x = Parameter(List(1.0, 2.0))
    val y = x * x
    assert(y.eval(gc)(sc) == List(1.0, 4.0))

    val z = y.grad(gc, x).get(sc)
    assert(z == List(List(2.0, 0.0), List(0.0, 4.0)))
  }

  it should "calculate derivative of a reciprocal" in {
    val x = Parameter[Id, Any](2.0)
    val y = Constant(1.0) / x
    assert(y.eval(gc)(sc) == 0.5)

    val z: Double = y.grad(gc, x).get(sc)
    assert(z == -0.25)
  }

  it should "calculate derivative of a scalar function" in {
    val x = Parameter[Id, Any](2.0)
    val y = log(x)
    assert(y.eval(gc)(sc) == scala.math.log(2.0))

    val z: Double = y.grad(gc, x).get(sc)
    assert(z == 0.5)
  }

  it should "calculate derivative of a function applied to a list" in {
    val x = Parameter(List(1.0, 2.0))
    val y = log(x)
    assert(y.eval(gc)(sc) == List(0.0, scala.math.log(2.0)))

    val z = y.grad(gc, x).get(sc)
    assert(z == List(List(1.0, 0.0), List(0.0, 0.5)))
  }

  it should "calculate probability of observation" in {
    val mu = Parameter[List, Int](List(0.0, 0.0))
    val sigma = Parameter[List, Int](List(1.0, 1.0))

    val normal = Normal(mu, sigma)
    val observation = normal.observe(List(1.0, 2.0))
    println(observation.logp.eval(gc)(sc))
  }

  it should "be able to implement linear regression" in {
//    val a = Normal[Id, Any](0.0, 1.0).sample
    val a = Parameter[Id, Any](0.0)
    val b = Normal[List, Int](List(0.0, 0.0), List(1.0, 1.0)).sample
    val e = Normal[Id, Any](1.0, 1.0).sample

    val data = List(
      (List(1.0, 2.0), 0.5),
      (List(2.0, 1.0), 1.0)
    )

    val observations = data.map {
      case (x, y) ⇒
        val s = a + sum(x * b)
        Normal(s, e).observe(y)
    }
    val logp = sum(observations.map(_.logp)) +
      b.logp + e.logp
    println(logp.eval(gc)(sc))

    val g_a: Double = logp.grad(gc, a).get(sc)
    println(g_a)
  }

  it should "calculate exp derivative" in {
    val x = Parameter[Id, Any](1.0)
    val y = exp(x)
    assert(y.eval(gc)(sc) == scala.math.exp(1.0))

    val z: Double = y.grad(gc, x).get(sc)
    assert(z == scala.math.exp(1.0))
  }

  it should "derive zero for gradient of constant" in {
    val x = Parameter[Id, Any](1.0)
    val y = Constant(2.0)
    val g: Option[_] = y.grad(gc, x)
    assert(g.isEmpty)
  }
}
