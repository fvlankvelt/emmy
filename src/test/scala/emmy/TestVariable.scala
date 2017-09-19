package emmy

import emmy.autodiff.{ContainerOps, EvaluationContext, ValueOps, Variable}

case class TestVariable[U[_], V, S](value: U[V])
                                   (implicit
                                    val vo: ValueOps[U, V, S],
                                    val ops: ContainerOps.Aux[U, S])
  extends Variable[U, V, S] {

  override def shape = ops.shapeOf(value)

  override def apply(ec: EvaluationContext[V]) = value

  override def logp() = ???

  override implicit val vt = vo.bind(ops.shapeOf(value))
}
