package emmy

import emmy.autodiff.{ ContainerOps, ContinuousVariable, Evaluable, EvaluationContext, Floating, ScalarOps, ValueOps }

import scalaz.Scalaz.Id

case class TestVariable[U[_], S](value: U[Double])(implicit val ops: ContainerOps.Aux[U, S])
  extends ContinuousVariable[U, S] {

  override def apply(ec: EvaluationContext) = value

  override def logp() = ???

  override val vt = Evaluable.fromConstant(ValueOps(Floating.doubleFloating, ops, ops.shapeOf(value)))

  override val so = ScalarOps.liftBoth[U, Double, Double](ScalarOps.doubleOps, ops)
}

object TestVariable {

  def apply(value: Double): TestVariable[Id, Any] = {
    TestVariable[Id, Any](value)
  }
}
