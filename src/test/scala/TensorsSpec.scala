import org.scalatest.FlatSpec
import pp.tensor._

import scala.language.higherKinds

class TensorsSpec extends FlatSpec {

  "ed" should "have a reasonable type" in {
    val ed = Domain.join(Domain(2), Domain(3))
    println(ed.sizes)

    val a = Domain[Succ[Zero]](Seq(1))
    val b = Domain[Succ[Succ[Zero]]](Seq(1))
    val c: Domain[Nat._1] = a
    assert(ed == Domain[Nat._2](sizes = List(2, 3)))
  }

  "variable" should "have covariant gradient" in {
    val a = Variable[Nat._1](Domain(1))
    val b = Variable[Nat._1](Domain(2))
    val t = Variable[Nat._1](Domain(3))
    val c = a.grad(a)
    val d: Expression[Nat._1, Nat._1] = a.grad(b)
    val x: Expression[Nat._2, Nat._1] = (a outer b) grad t
  }

}
