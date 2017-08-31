import org.scalatest.FlatSpec
import pp.{Var, ad}
import ad._

import scalaz._
import scalaz.Scalaz._

class AdSpec extends FlatSpec {

  "AD" should "calculate scalar derivative" in {
    val x = Var[Id, Double](2.0)
    val y = x * x
    val z: Double = y.grad(x)
    assert(z == 4.0)
  }

  it should "calculate vector derivative on List" in {
    val x = Var[List, Double](List(1.0, 2.0))
    val y = x * x
    val z = y.grad(x)
    assert(z == List(List(2.0, 0.0), List(0.0, 4.0)))
  }
}
