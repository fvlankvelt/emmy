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
    print(z)
  }
}
