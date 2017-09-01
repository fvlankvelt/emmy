import org.scalatest.FlatSpec
import pp.ad._

import scalaz._
import scalaz.Scalaz._

class AdSpec extends FlatSpec {

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
    print(observation.logp()())
  }
}
