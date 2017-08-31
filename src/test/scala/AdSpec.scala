import org.scalatest.FlatSpec
import pp.{Var, ad}
import ad._

import scalaz._
import scalaz.Scalaz._

class AdSpec extends FlatSpec {

  "AD" should "calculate scalar derivative" in {
    val x = Var[Id, Double](2.0)
    val y = x * x
    assert(y() == 4.0)

    val z: Double = y.grad(x)
    assert(z == 4.0)
  }

  it should "calculate vector derivative on List" in {
    val x = Var[List, Double](List(1.0, 2.0))
    val y = x * x
    assert(y() == List(1.0, 4.0))

    val z = y.grad(x)
    assert(z == List(List(2.0, 0.0), List(0.0, 4.0)))
  }

  it should "calculate derivative of a scalar function" in {
    val x = Var[Id, Double](2.0)
    val y = pp.log(x)
    assert(y() == scala.math.log(2.0))

    val z: Double = y.grad(x)
    assert(z == 0.5)
  }

  it should "calculate derivative of a function applied to a list" in {
    val x = Var[List, Double](List(1.0, 2.0))
    val y = pp.log(x)
    assert(y() == List(0.0, scala.math.log(2.0)))

    val z = y.grad(x)
    assert(z == List(List(1.0, 0.0), List(0.0, 0.5)))
  }
}
