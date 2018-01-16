package emmy.distribution

import emmy.autodiff.{ ConstantLike, Expression, Node, Variable, Visitor }

import scalaz.Scalaz.Id

trait Distribution[U[_], V, S] {

  def sample: Variable[U, V, S]

  def observe(data: U[V]): Observation[U, V, S]
}

trait Factor extends Node {

  override def visit[R](visitor: Visitor[R]): R = {
    visitor.visitFactor(this)
  }

  def logp: Expression[Id, Double, Any]
}

trait Observation[U[_], V, S] extends ConstantLike[U, V, S] with Factor {

  override def visit[R](visitor: Visitor[R]): R = {
    visitor.visitObservation(this)
  }

}
