package emmy

import emmy.autodiff.{ContainerOps, Evaluable, EvaluationContext, Floating, ScalarOps, ValueOps, Variable}

case class TestVariable[U[_], S](value: U[Double])
                                   (implicit
                                    val ops: ContainerOps.Aux[U, S])
  extends Variable[U, S] {

  override def apply(ec: EvaluationContext) = value

  override def logp() = ???

  override val vt = Evaluable.fromConstant(ValueOps(Floating.doubleFloating, ops, ops.shapeOf(value)))

  override val so = ScalarOps.liftBoth[U, Double, Double](ScalarOps.doubleOps, ops)
}
