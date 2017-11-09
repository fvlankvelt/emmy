package emmy.distribution

import emmy.autodiff.{ConstantLike, Expression, Variable, Visitor}

import scalaz.Scalaz.Id

trait Distribution[U[_], V, S] {

  def sample: Expression[U, V, S]

  def observe(data: U[V]): Observation[U, V, S]
}

trait Factor {

  def logp(): Expression[Id, Double, Any]
}

trait Observation[U[_], V, S] extends ConstantLike[U, V, S] with Factor {

  override def visit[R](visitor: Visitor[R]): R = {
    visitor.visitObservation(this)
  }

}
