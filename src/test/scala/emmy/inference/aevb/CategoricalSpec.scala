package emmy.inference.aevb

import breeze.numerics.abs
import emmy.autodiff.ContainerOps.Aux
import emmy.autodiff._
import emmy.distribution.Factor
import org.scalatest.FlatSpec

import scalaz.Scalaz
import scalaz.Scalaz.Id

class CategoricalSpec extends FlatSpec {

  case class TestCatVariable()(implicit val ops: ContainerOps.Aux[Id, Any])
    extends CategoricalVariable {

    var value: Int = 0

    override def K: Evaluable[Int] = Evaluable.fromConstant(2)

    override def eval(ec: GradientContext) = value

    override def logp = Constant(0.0)

    override val vt = Evaluable.fromConstant(ValueOps(Floating.intFloating, ops, ops.shapeOf(value)))

    override val so = ScalarOps.intDoubleOps

  }

  /**
    * Verify that categorical variables are sampled correctly in the optimization process.
    */
  "The Categorical sampler" should "determine thetas to the exact solution" in {
    val p_zero = 0.87
    val variable = TestCatVariable()
    val observationLogp = new Expression[Id, Double, Any] {
      override implicit val ops: Aux[Scalaz.Id, Shape] =
        ContainerOps.idOps

      override implicit val so: ScalarOps[Scalaz.Id[Double], Scalaz.Id[Double]] =
        ScalarOps.doubleOps

      override implicit def vt: Evaluable[ValueOps[Scalaz.Id, Double, Any]] =
        ValueOps.idOps(Floating.doubleFloating)

      // posterior logs - should be matched by thetas of categorical dist
      val logs = Seq(math.log(p_zero), math.log(1.0 - p_zero))

      override def eval(ec: GradientContext): Evaluable[Double] = {
        ec(variable).map {
          index => logs(index)
        }
      }
    }

    class TestObservation extends Factor {
      override def logp: Expression[Scalaz.Id, Double, Any] = observationLogp
    }

    val model = AEVBModel(Seq(variable: Node))
    val newModel = model.update(Range(0, 1).map { _ => new TestObservation })
    val newVars = newModel.variables
    val params = newVars.head
      .parameters.head
      .asInstanceOf[ParameterHolder[IndexedSeq, Int]]
    val value = params.value.get
    val thetas = value.map{math.exp}
    assert(abs(math.log(thetas.head / thetas.sum) - math.log(p_zero)) < 0.15)
  }

}
