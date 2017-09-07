package emmy.autodiff

import emmy.autodiff.ContainerOps.Aux
import emmy.distribution.Normal
import org.scalatest.FlatSpec

import scalaz.Scalaz._

class AutoDiffSpec extends FlatSpec {

  val gc = new GradientContext {

    override def apply[U[_], V, S](n: Node[U, V, S]) = {
      n match {
        case v : TestVariable[U, V, S] => v.value
        case _ => n.apply(this)
      }
    }

    override def apply[W[_], U[_], V, T, S](n: Node[U, V, S], v: Variable[W, V, T])(implicit wOps: Aux[W, T]) = {
      n.grad(this, v)
    }
  }

  val ec : EvaluationContext = gc

  case class TestVariable[U[_], V, S](value: U[V])
                                     (implicit
                                      val vo: ValueOps[U, V, S],
                                      val ops: ContainerOps.Aux[U, S])
    extends Variable[U, V, S] {

    override def shape = ops.shapeOf(value)

    override def apply(ec: EvaluationContext) = value

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
    val a = Normal(
      Constant(0.0),
      Constant(1.0)
    ).sample

    val b = Normal(
      Constant(List(0.0, 0.0)),
      Constant(List(1.0, 1.0))
    ).sample

    val e = Normal(
      Constant(1.0),
      Constant(1.0)
    ).sample

    val data = List(
      (List(1.0, 2.0), 0.5),
      (List(2.0, 1.0), 1.0)
    )

    val observations = data.map {
      case (x, y) =>
        val cst = Constant(x)
        val s = a + sum(cst * b)
        Normal(s, e).observe(y)
    }
    val logp = observations.map(_.logp()).sum +
      a.logp() + b.logp() + e.logp()
    println(logp(ec))

    val g_a: Double = logp.grad(gc, a)
    println(g_a)
  }

  /*
  it should "update variational parameters for each (minibatch of) data point(s)" in {
    val data = List(0.2, 1.0, 0.5)

    val mu = Normal(Constant(0.0), Constant(1.0)).sample
    val sigma = Normal(Constant(1.0), Constant(0.5)).sample
    val dist = Normal(mu, sigma)

    val initialModel = new Model {override def update[U[_], V, S](o: Observation[U, V, S]) = ???

      override def sample() = ???

      override def addVariable[U[_], V, S](variable: Variable[U, V, S]) = ???
    }

    val newModel = data.foldLeft(initialModel) {
      case (m, d) =>
        val observation = dist.observe(d)
        m.update(observation)
    }
  }
  */
}
