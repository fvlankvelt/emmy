package emmy

import emmy.autodiff.{ContainerOps, Evaluable, EvaluationContext, Floating, ValueOps, Variable}

case class TestVariable[U[_], V, S](value: U[V])
                                   (implicit
                                    val fl: Floating[V],
                                    val ops: ContainerOps.Aux[U, S])
  extends Variable[U, V, S] {

  override def apply(ec: EvaluationContext[V]) = value

  override def logp() = ???

  override val vt = Evaluable.fromConstant(ValueOps(fl, ops, ops.shapeOf(value)))
}
