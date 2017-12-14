package emmy.autodiff

import scalaz.Scalaz.Id

trait ConstantLike[U[_], V, S] extends Expression[U, V, S] {

  def value: Evaluable[U[V]]

  override def eval(ec: GradientContext): Evaluable[U[V]] =
    value

  override def toString: String =
    value.toString
}

case class Constant[U[_], V, S](value: Evaluable[U[V]])(implicit
    val fl: Floating[V],
                                                        val so:  ScalarOps[U[Double], U[V]],
                                                        val ops: ContainerOps.Aux[U, S]
)
  extends ConstantLike[U, V, S] {

  override val vt: Evaluable[ValueOps[U, V, S]] =
    value.map(toVT)

  private def toVT(v: U[V]) = {
    val shape = ops.shapeOf(v)
    ValueOps(fl, ops, shape)
  }
}

object Constant {

  def apply[U[_], V, S](value: U[V])(implicit
    fl: Floating[V],
                                     so:  ScalarOps[U[Double], U[V]],
                                     ops: ContainerOps.Aux[U, S]
  ): Constant[U, V, S] =
    Constant(Evaluable.fromConstant(value))

  def apply(value: Double): Constant[Id, Double, Any] = {
    Constant[Id, Double, Any](value)(Floating.doubleFloating, ScalarOps.doubleOps, ContainerOps.idOps)
  }
}

class Parameter[U[_], S](var v: Evaluable[U[Double]])(implicit
    fl: Floating[Double],
                                                      val so:  ScalarOps[U[Double], U[Double]],
                                                      val ops: ContainerOps.Aux[U, S]
)
  extends ConstantLike[U, Double, S] {

  override def value = {
    val self = this
    ctx ⇒ {
      val value = v(ctx)
      //      println(s"Param(${self.hashCode}): $value")
      value
    }
  }

  override def visit[R](visitor: Visitor[R]): R = {
    visitor.visitParameter(this)
  }

  override val vt: Evaluable[ValueOps[U, Double, S]] =
    value.map(toVT)

  private def toVT(v: U[Double]) = {
    val shape = ops.shapeOf(v)
    ValueOps(fl, ops, shape)
  }

  override def grad[W[_], T](gc: GradientContext, v: Parameter[W, T]) = {
    if (this == v) {
      val value = gc(v)
      val wOps = v.ops
      Some { ctx ⇒
        val valT = vt(ctx)
        val ev = value(ctx)
        val shape = wOps.shapeOf(ev)
        wOps.eye(shape, valT.valueVT.one, valT.valueVT.zero).asInstanceOf[W[U[Double]]]
      }
    }
    else {
      None
    }
  }

  override def toString: String = s"param#$hashCode"
}
