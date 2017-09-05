package emmy.autodiff

import org.scalatest.FlatSpec

import scalaz.Scalaz._

class AutoDiffSpec extends FlatSpec {

  "AD" should "calculate scalar derivative" in {
    val x = Variable[Id, Double, Any](2.0)
    val y = x * x
    assert(y() == 4.0)

    val z: Double = y.grad(x)
    assert(z == 4.0)
  }

  it should "calculate vector derivative on List" in {
    val x = Variable[List, Double, Int](List(1.0, 2.0))
    val y = x * x
    assert(y() == List(1.0, 4.0))

    val z = y.grad(x)
    assert(z == List(List(2.0, 0.0), List(0.0, 4.0)))
  }

  it should "calculate derivative of a scalar function" in {
    val x = Variable[Id, Double, Any](2.0)
    val y = log(x)
    assert(y() == scala.math.log(2.0))

    val z: Double = y.grad(x)
    assert(z == 0.5)
  }

  it should "calculate derivative of a function applied to a list" in {
    val x = Variable[List, Double, Int](List(1.0, 2.0))
    val y = log(x)
    assert(y() == List(0.0, scala.math.log(2.0)))

    val z = y.grad(x)
    assert(z == List(List(1.0, 0.0), List(0.0, 0.5)))
  }

  it should "calculate probability of observation" in {
    val mu = Variable[List, Double, Int](List(0.0, 0.0))
    val sigma = Variable[List, Double, Int](List(1.0, 1.0))

    val normal = Normal(mu, sigma)
    val observation = normal.observe(List(1.0, 2.0))
    println(observation.logp()())
  }

  it should "be able to implement linear regression" in {
    implicit val model = new Model {
      override def valueOf[U[_], V, S](v: Variable[U, V, S])(implicit vo: ValueOps[U, V, S], ops: ContainerOps.Aux[U, S]) = {
        vo.bind(v.shape).rnd
      }
    }
    val a = Normal[Id, Double, Any](Constant[Id, Double, Any](0.0), Constant[Id, Double, Any](1.0)).sample
    val b = Normal[List, Double, Int](Constant(List(0.0, 0.0)), Constant(List(1.0, 1.0))).sample
    val e = Normal[Id, Double, Any](Constant[Id, Double, Any](1.0), Constant[Id, Double, Any](1.0)).sample

    val data = List(
      (List(1.0, 2.0), 0.5),
      (List(2.0, 1.0), 1.0)
    )

    val s = data.map {
      case (x, y) =>
        val cst = Constant[List, Double, Int](x)
        val s = a + sum(cst * b)
        Normal(s, e).observe(y)
    }
    val logp = s.map(_.logp()).sum + a.logp() + b.logp() + e.logp()
    println(logp())

    val g_a: Double = logp.grad(a)
    println(g_a)
  }
}
