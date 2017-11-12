package emmy.inference.aevb

import breeze.numerics.abs
import emmy.autodiff.ContainerOps.Aux
import emmy.autodiff.{ CategoricalVariable, ContainerOps, Evaluable, EvaluationContext, Expression, Floating, Node, ScalarOps, ValueOps }
import emmy.inference.ModelGradientContext
import org.scalatest.FlatSpec

import scalaz.Scalaz
import scalaz.Scalaz.Id

class CategoricalSamplerSpec extends FlatSpec {

  case class TestCatVariable()(implicit val ops: ContainerOps.Aux[Id, Any])
    extends CategoricalVariable {

    var value: Int = 0

    override def K: Evaluable[Int] = Evaluable.fromConstant(2)

    override def apply(ec: EvaluationContext) = value

    override def logp() = ???

    override val vt = Evaluable.fromConstant(ValueOps(Floating.intFloating, ops, ops.shapeOf(value)))

    override val so = ScalarOps.intDoubleOps

  }

  "The Categorical sampler" should "determine thetas to the exact solution" in {
    val p_zero = 0.87
    val variable = TestCatVariable()
    val logp = new Expression[Id, Double, Any] {
      override implicit val ops: Aux[Scalaz.Id, Shape] =
        ContainerOps.idOps

      override implicit val so: ScalarOps[Scalaz.Id[Double], Scalaz.Id[Double]] =
        ScalarOps.doubleOps

      override implicit def vt: Evaluable[ValueOps[Scalaz.Id, Double, Any]] =
        ValueOps.idOps(Floating.doubleFloating)

      // posterior logs - should be matched by thetas of categorical dist
      val logs = Seq(math.log(p_zero), math.log(1.0 - p_zero))

      override def apply(ec: EvaluationContext): Scalaz.Id[Double] = {
        val index = ec(variable)
        logs(index)
      }
    }

    {
      val pLogP = logp.logs.map { lp ⇒ lp * math.exp(lp) }.sum
      println(s"Expected offset: $pLogP")
    }

    val sampler = new CategoricalSampler(variable, Array(0.5, 0.5))
    val newSampler = (0 until 200).foldLeft(sampler) {
      case (s, idx) ⇒
        val model = new AEVBSamplersModel(Map((variable: Node) -> s))
        val gc = new ModelGradientContext(model)
        val update = s.update(logp, gc, 1.0 / (idx + 1))
        //        if (idx % 80 == 0) {
        //          println()
        //        }
        //        println(s"delta: ${update._2}")
        update._1
    }
    // 3/2, 2/3
    // (3/2) / (2/3) = 9/4 ~ 2.25
    assert(abs(newSampler.thetas(0) - p_zero) < 0.10)
    //    assert(abs(newSampler.mu - 1.0) < 0.01)
  }

}
