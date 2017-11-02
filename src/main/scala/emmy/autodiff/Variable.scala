package emmy.autodiff

import emmy.distribution.Factor

import scalaz.Scalaz.Id

sealed trait Variable[U[_], V, S] extends Expression[U, V, S] with Factor

trait ContinuousVariable[U[_], S] extends Variable[U, Double, S] {

  override def visit[R](visitor: Visitor[R]): R = {
    visitor.visitContinuousVariable(this)
  }

  override def grad[W[_], T](gc: GradientContext, v: ContinuousVariable[W, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val shape = wOps.shapeOf(gc(v))
    val valT = vt(gc)
    if (this == v) {
      wOps.eye(shape, valT.valueVT.one, valT.valueVT.zero).asInstanceOf[Gradient[W, U]]
    }
    else {
      wOps.fill(shape, valT.zero)
    }
  }
}

trait CategoricalVariable extends Variable[Id, Int, Any] {

  def K: Evaluable[Int]

  override def visit[R](visitor: Visitor[R]): R = {
    visitor.visitCategoricalVariable(this)
  }

}

