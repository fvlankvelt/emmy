package emmy.autodiff

import emmy.distribution.Factor

import scalaz.Scalaz.Id

sealed trait Variable[U[_], V, S] extends Expression[U, V, S] with Factor

trait ContinuousVariable[U[_], S] extends Variable[U, Double, S] {

  override def visit[R](visitor: Visitor[R]): R = {
    visitor.visitContinuousVariable(this)
  }

}

trait CategoricalVariable extends Variable[Id, Int, Any] {

  def K: Evaluable[Int]

  override def visit[R](visitor: Visitor[R]): R = {
    visitor.visitCategoricalVariable(this)
  }

}

