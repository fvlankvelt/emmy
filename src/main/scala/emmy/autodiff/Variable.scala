package emmy.autodiff

import scala.collection.mutable


trait Variable[U[_], V, S] extends Node[U, V, S] {

  private val cache = mutable.Buffer.newBuilder[(AnyRef, Any)].result()

  def get[W[_], T](node: Node[W, V, T])(fn: => Gradient[U, W, V]): Gradient[U, W, V] = {
    cache.find(pair => pair._1 eq node) match {
      case Some((_, grad)) =>
        grad.asInstanceOf[Gradient[U, W, V]]
      case _ =>
        val grad: Gradient[U, W, V] = fn
        cache.append((node, grad))
        grad
    }
  }

  override def calcGrad[W[_], T](v: Variable[W, V, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val ops = implicitly[ContainerOps[W]]
    val shape = ops.shapeOf(v())
    if (this == v) {
      ops.eye(shape, vt.valueVT.one, vt.valueVT.zero).asInstanceOf[Gradient[W, U, V]]
    } else {
      ops.fill(shape, vt.zero)
    }
  }
}

object Variable {

  def apply[U[_], V, S](v: U[V])
                       (implicit
                        valueType: ValueOps[U, V, S],
                        cOps: ContainerOps.Aux[U, S]): Variable[U, V, S] =
    new Variable[U, V, S] {

      override val shape = cOps.shapeOf(v)

      override implicit val vt = valueType.bind(shape)

      override implicit val ops = cOps

      override def value = v
    }
}
