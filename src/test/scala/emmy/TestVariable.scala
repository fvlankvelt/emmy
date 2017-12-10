package emmy

import emmy.autodiff.{ Constant, ContainerOps, ContinuousVariable, Evaluable, GradientContext, Floating, ScalarOps, ValueOps }

import scalaz.Scalaz.Id

class TestVariable[U[_], S](val value: U[Double])(implicit val ops: ContainerOps.Aux[U, S])
  extends ContinuousVariable[U, S] {

  private val id = TestVariable.newId()

  override def eval(ec: GradientContext) = value

  override val logp = Constant(0.0)

  override val vt = Evaluable.fromConstant(ValueOps(Floating.doubleFloating, ops, ops.shapeOf(value)))

  override val so = ScalarOps.liftBoth[U, Double, Double](ScalarOps.doubleOps, ops)

  override def toString = s"var#$id($value)"
}

object TestVariable {

  var counter: Int = 0

  def newId(): Int = {
    counter += 1
    counter
  }

  def apply(value: Double): TestVariable[Id, Any] = {
    new TestVariable[Id, Any](value)
  }
}
