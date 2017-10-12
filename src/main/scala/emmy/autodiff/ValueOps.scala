package emmy.autodiff

import scalaz.Scalaz.Id


case class ValueOps[U[_], V, Shape](
                                     valueVT: Floating[V],
                                     ops: ContainerOps.Aux[U, Shape],
                                     shape: Shape
                                   ) extends Floating[U[V]] {

  def forDouble: ValueOps[U, Double, Shape] = ValueOps(Floating.doubleFloating, ops, shape)

  override def sqrt = new UnaryValueFunc[U[V]] {

    val name = "sqrt"

    private val upstream = valueVT.sqrt

    override def grad(v: U[V]) = ops.map(v)(upstream.grad)

    override def apply(v: U[V]) = ops.map(v)(upstream.apply)

  }

  override def log = new UnaryValueFunc[U[V]] {

    val name = "log"

    private val upstream = valueVT.log

    override def grad(v: U[V]) = ops.map(v)(upstream.grad)

    override def apply(v: U[V]) = ops.map(v)(upstream.apply)
  }

  override def exp = new UnaryValueFunc[U[V]] {

    val name = "exp"

    private val upstream = valueVT.exp

    override def grad(v: U[V]) = ops.map(v)(upstream.grad)

    override def apply(v: U[V]) = ops.map(v)(upstream.apply)
  }

  override def lgamma = new UnaryValueFunc[U[V]] {

    val name = "lgamma"

    private val upstream = valueVT.lgamma

    override def grad(v: U[V]) = ops.map(v)(upstream.grad)

    override def apply(v: U[V]) = ops.map(v)(upstream.apply)
  }

  override def tanh = new UnaryValueFunc[U[V]] {

    val name = "tanh"

    private val upstream = valueVT.tanh

    override def grad(v: U[V]) = ops.map(v)(upstream.grad)

    override def apply(v: U[V]) = ops.map(v)(upstream.apply)
  }

  override def sum = new CollectValueFunc[U[V]] {

    val name = "sum"

    private val upstream = valueVT.sum

    override def start = null.asInstanceOf[U[V]]

    override def apply(acc: U[V], v: U[V]) = ops.zipMap(acc, v) {
      (a, x) => upstream(if (a == null) upstream.start else a, x)
    }

    override def grad(a: U[V], v: U[V]) = ops.zipMap(a, v)(upstream.grad)
  }

  override def div(x: U[V], y: U[V]) = ops.zipMap(x, y)(valueVT.div)

  override def plus(x: U[V], y: U[V]) = ops.zipMap(x, y)(valueVT.plus)

  override def minus(x: U[V], y: U[V]) = ops.zipMap(x, y)(valueVT.minus)

  override def times(x: U[V], y: U[V]) = ops.zipMap(x, y)(valueVT.times)

  override def negate(x: U[V]) = ops.map(x)(valueVT.negate)

  override def abs(x: U[V]): U[V] = ops.map(x)(valueVT.abs)

  override def fromInt(x: Int) = ops.fill(shape, valueVT.fromInt(x))

  override def rnd = ops.fill(shape, valueVT.rnd)

  override def toInt(x: U[V]) = invalidOp("toInt")

  override def toLong(x: U[V]) = invalidOp("toLong")

  override def toFloat(x: U[V]) = invalidOp("toFloat")

  override def toDouble(x: U[V]) = invalidOp("toDouble")

  override def compare(x: U[V], y: U[V]) = invalidOp("compare")

  private def invalidOp[R](op: String): R = {
    throw new UnsupportedOperationException(s"Operation '$op' is not valid on unbound ValueOps")
  }
}

object ValueOps {
  implicit def idOps[V](implicit fl: Floating[V]): ValueOps[Id, V, Any] =
    ValueOps(fl, ContainerOps.idOps, null)
}